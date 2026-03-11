#!/usr/bin/env python3
"""
Todo-List Planning Agent for ALFWorld.

This agent uses code-tracked state (visited locations, inventory, unvisited
locations) injected into the prompt.  

The LLM decides what to do next (todo_list + next_action).

Usage:
    export TRITONAI_API_KEY="your-api-key-here"
    python scripts/todo_agent.py configs/base_config.yaml --num_games 20
"""

import os
import sys
import json
import re
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import yaml
import requests

from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are a planner for an agent in a text-based household environment (ALFWorld).

Each turn you receive:
- The GOAL you must achieve
- Ground-truth STATE
- The latest observation from the environment
- Admissible commands you can use

You must return a JSON object with exactly these keys:
{
  "todo_list": ["<subgoal 1>", "<subgoal 2>", ...],
  "next_action": "<exact action string from admissible commands>"
}

Rules:
1. "next_action" MUST be one of the admissible commands — copy it EXACTLY.
2. "todo_list" should contain short, concrete subgoals (max 5).
   Mark completed subgoals by removing them.
   "todo_list" must NEVER be empty until the goal is fully achieved.
3. Return ONLY the JSON object. No other text, no markdown fences.

Strategy:
- ONLY pick up the object required by the goal. You can hold ONE item at a time.
  Picking up the wrong object blocks all progress.
- To find an object: systematically visit locations you have NOT checked yet.
  Prioritise countertops and tables (food is usually there).
- To heat an object: hold it, go to microwave, use "heat <obj> with microwave 1".
- To cool an object: hold it, go to fridge, use "cool <obj> with fridge 1".
- To clean an object: hold it, go to sinkbasin, use "clean <obj> with sinkbasin 1".
- To examine under light: hold the object, go to the lamp, use "use desklamp 1".
- NEVER revisit a location you already checked unless you need to interact with
  something there (e.g. put an object down).
- If you have checked many locations and haven't found the target, try the
  remaining UNVISITED locations listed in STATE."""


STEP_USER_TEMPLATE = """\
GOAL: {goal}

STATE (code-tracked, always accurate):
  Current location: {current_location}
  Holding: {holding}
  Visited ({n_visited}/{n_total} locations): {visited_summary}
  NOT yet visited: {unvisited}

Observation: {observation}
Admissible commands: {admissible_commands}

Respond with ONLY a JSON object: {{"todo_list": [...], "next_action": "..."}}"""


# ---------------------------------------------------------------------------
# Observation parser — extracts structured state from env text
# ---------------------------------------------------------------------------

_RE_ARRIVE = re.compile(
    r"You arrive at (.+?)\."
)
_RE_ON_SURFACE = re.compile(
    r"(?:On|In) (?:the )?(.+?), you see (.+)\."
)
_RE_NOTHING = re.compile(
    r"(?:On|In) (?:the )?(.+?), you see nothing"
)
_RE_PICKUP = re.compile(
    r"You pick up the (.+?) from"
)
_RE_PUT = re.compile(
    r"You (?:put|move) the (.+?) (?:in/on|to|in|on) "
)
_RE_OPEN = re.compile(
    r"You open the (.+?)\."
)
_RE_GOTO = re.compile(
    r"^go to (.+)$"
)


def _extract_target_object(goal: str) -> str:
    """Guess the target object type from the goal text.

    E.g. 'heat some apple and put it in fridge.' -> 'apple'
         'put a mug in desk.' -> 'mug'
         'examine the alarmclock with the desklamp.' -> 'alarmclock'
         'clean some bowl and put it in cabinet.' -> 'bowl'
    """
    goal_lower = goal.lower().strip().rstrip(".")
    for pattern in [
        r"(?:heat|cool|clean)\s+(?:some|a|an|the)\s+(\w+)",
        r"(?:put)\s+(?:a cool|a hot|a clean|some|a|an|the)\s+(\w+)",
        r"(?:examine)\s+(?:the|a|an|some)\s+(\w+)",
        r"(?:put)\s+(?:some|a|an|the)\s+(\w+)",
    ]:
        m = re.search(pattern, goal_lower)
        if m:
            return m.group(1)
    return ""


def _all_go_to_locations(admissible_commands: List[str]) -> List[str]:
    """Extract location names from 'go to X' commands."""
    locs = []
    for cmd in admissible_commands:
        m = _RE_GOTO.match(cmd)
        if m:
            locs.append(m.group(1))
    return locs


# ---------------------------------------------------------------------------
# TodoAgent class
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = 20


class TodoAgent:
    """Agent with code-tracked state and LLM-based planning."""

    def __init__(self, api_key: str, api_url: str, model: str,
                 max_tokens: int = 512, temperature: float = 0.1):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.goal: str = ""
        self.target_object: str = ""
        self.todo_list: List[str] = []
        self.conversation_history: List[Dict] = []
        self.task_type: str = ""

        # Code-tracked state
        self.current_location: str = "middle of room"
        self.holding: str = "nothing"
        self.visited: Dict[str, str] = {}  # location -> what was found
        self.all_locations: List[str] = []  # set on first step from admissible cmds

    def reset(self, goal: str, task_type: str = ""):
        self.goal = goal
        self.target_object = _extract_target_object(goal)
        self.todo_list = []
        self.conversation_history = []
        self.task_type = task_type
        self.current_location = "middle of room"
        self.holding = "nothing"
        self.visited = {}
        self.all_locations = []

    # ----- state tracking from observations -----

    def _update_state(self, observation: str, action: str,
                      admissible_commands: List[str]):
        """Parse observation to update code-tracked state."""
        if not self.all_locations:
            self.all_locations = _all_go_to_locations(admissible_commands)

        m = _RE_ARRIVE.search(observation)
        if m:
            loc = m.group(1).strip()
            self.current_location = loc
            nothing = _RE_NOTHING.search(observation)
            surface = _RE_ON_SURFACE.search(observation)
            if nothing:
                self.visited[loc] = "nothing"
            elif surface:
                self.visited[loc] = surface.group(2).strip()
            else:
                if loc not in self.visited:
                    self.visited[loc] = "(closed/unknown)"

        m = _RE_OPEN.search(observation)
        if m:
            container = m.group(1).strip()
            surface = _RE_ON_SURFACE.search(observation)
            nothing = _RE_NOTHING.search(observation)
            if surface:
                self.visited[container] = surface.group(2).strip()
            elif nothing:
                self.visited[container] = "nothing"
            else:
                self.visited[container] = "empty"

        m = _RE_PICKUP.search(observation)
        if m:
            self.holding = m.group(1).strip()

        m = _RE_PUT.search(observation)
        if m:
            self.holding = "nothing"

        if "clean" in observation.lower() and "using" in observation.lower():
            pass
        if "heat" in observation.lower() and "using" in observation.lower():
            pass
        if "cool" in observation.lower() and "using" in observation.lower():
            pass

    def _unvisited(self) -> List[str]:
        return [loc for loc in self.all_locations if loc not in self.visited]

    def _state_summary(self) -> Dict[str, str]:
        visited_parts = []
        for loc, found in self.visited.items():
            visited_parts.append(f"{loc}: {found}")
        visited_str = "; ".join(visited_parts) if visited_parts else "none"

        unvisited = self._unvisited()
        unvisited_str = ", ".join(unvisited) if unvisited else "ALL locations checked"

        return {
            "current_location": self.current_location,
            "holding": self.holding,
            "visited_summary": visited_str,
            "unvisited": unvisited_str,
            "n_visited": str(len(self.visited)),
            "n_total": str(len(self.all_locations)),
        }

    # ----- action filtering -----

    def _filter_action(self, action: str,
                       admissible_commands: List[str]) -> str:
        """Block picking up irrelevant objects."""
        if not self.target_object:
            return action

        if action.startswith("take "):
            obj_part = action.split("from")[0].replace("take ", "").strip()
            obj_type = re.sub(r"\s*\d+$", "", obj_part).lower()
            if obj_type != self.target_object.lower():
                unvisited = self._unvisited()
                for loc in unvisited:
                    cmd = f"go to {loc}"
                    if cmd in admissible_commands:
                        print(f"    [filter] Blocked '{action}' (wrong object). "
                              f"Redirecting to '{cmd}'")
                        return cmd
                if "look" in admissible_commands:
                    print(f"    [filter] Blocked '{action}' (wrong object). "
                          f"Falling back to 'look'")
                    return "look"

        return action

    # ----- LLM interaction -----

    def plan_step(self, observation: str,
                  admissible_commands: List[str]) -> str:
        """Update state from observation, call LLM, filter action."""
        if not self.all_locations:
            self.all_locations = _all_go_to_locations(admissible_commands)

        state = self._state_summary()

        user_content = STEP_USER_TEMPLATE.format(
            goal=self.goal,
            observation=observation,
            admissible_commands=json.dumps(admissible_commands),
            **state,
        )

        self.conversation_history.append(
            {"role": "user", "content": user_content}
        )
        self._trim_history()

        messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
        messages.extend(self.conversation_history)

        response_text = self._call_llm(messages)
        action = self._parse_and_update(response_text, admissible_commands)

        action = self._filter_action(action, admissible_commands)

        self.conversation_history.append(
            {"role": "assistant", "content": response_text}
        )

        self._update_state(observation, action, admissible_commands)

        return action

    def _trim_history(self):
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.conversation_history) > max_messages:
            first_msg = self.conversation_history[0]
            self.conversation_history = (
                [first_msg]
                + self.conversation_history[-(max_messages - 1):]
            )

    def _call_llm(self, messages: List[Dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except (requests.RequestException, KeyError, IndexError) as e:
                print(f"  [API error attempt {attempt + 1}/3]: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        return '{"todo_list": [], "next_action": "look"}'

    def _parse_and_update(self, response_text: str,
                          admissible_commands: List[str]) -> str:
        parsed = self._extract_json(response_text)

        if parsed is None:
            print(f"  [parse error] Could not extract JSON. Raw response:\n    {response_text[:300]}")
            return self._fallback_action(admissible_commands)

        if "todo_list" in parsed and isinstance(parsed["todo_list"], list):
            new_todos = [str(t) for t in parsed["todo_list"] if t]
            self.todo_list = new_todos[:5]

        action = parsed.get("next_action", "")
        action = self._validate_action(action, admissible_commands)
        return action

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _validate_action(self, action: str,
                         admissible_commands: List[str]) -> str:
        if not action:
            return self._fallback_action(admissible_commands)

        if action in admissible_commands:
            return action

        action_lower = action.lower().strip()
        for cmd in admissible_commands:
            if cmd.lower().strip() == action_lower:
                return cmd

        for cmd in admissible_commands:
            if cmd.lower() in action_lower:
                return cmd

        for cmd in admissible_commands:
            if action_lower in cmd.lower():
                return cmd

        return self._fallback_action(admissible_commands)

    def _fallback_action(self, admissible_commands: List[str]) -> str:
        if self.holding != "nothing":
            for cmd in admissible_commands:
                if cmd.startswith("put ") or cmd.startswith("move "):
                    return cmd
                if cmd.startswith("heat ") or cmd.startswith("cool ") or cmd.startswith("clean "):
                    return cmd
                if cmd == "use desklamp 1":
                    return cmd
        unvisited = self._unvisited()
        for loc in unvisited:
            cmd = f"go to {loc}"
            if cmd in admissible_commands:
                return cmd
        if "look" in admissible_commands:
            return "look"
        return admissible_commands[0] if admissible_commands else "look"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_task_type(gamefile_path: str) -> str:
    for task_type in TASK_TYPES.values():
        if task_type in gamefile_path:
            return task_type
    return "unknown"


def load_games(game_files: List[str], num_games: int) -> List[str]:
    files = sorted(game_files)
    if num_games <= 0 or num_games >= len(files):
        return files

    groups: Dict[str, List[str]] = {}
    for f in files:
        tt = detect_task_type(f)
        groups.setdefault(tt, []).append(f)

    types = sorted(groups.keys())
    per_type = num_games // len(types)
    remainder = num_games % len(types)

    selected = []
    for i, tt in enumerate(types):
        take = per_type + (1 if i < remainder else 0)
        selected.extend(groups[tt][:take])

    return sorted(selected)


def extract_goal(observation: str) -> str:
    if "Your task is to:" in observation:
        return observation.split("Your task is to:")[1].split("\n")[0].strip()
    return observation.strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_todo_agent(config: Dict, num_games: int, max_steps: int,
                   api_key: str, api_url: str, model: str,
                   eval_split: str, output_file: Optional[str]):

    env_type = config["env"]["type"]
    AlfredEnvClass = get_environment(env_type)
    alfred_env = AlfredEnvClass(config, train_eval=eval_split)
    alfred_env.game_files = load_games(alfred_env.game_files, num_games)
    alfred_env.num_games = len(alfred_env.game_files)

    print(f"\nEvaluating on {alfred_env.num_games} games (split={eval_split})")

    env = alfred_env.init_env(batch_size=1)

    agent = TodoAgent(
        api_key=api_key,
        api_url=api_url,
        model=model,
    )

    results = []
    task_successes: Dict[str, List[bool]] = defaultdict(list)

    for game_idx in range(alfred_env.num_games):
        obs, infos = env.reset()
        observation = obs[0]
        admissible = list(infos["admissible_commands"][0])
        gamefile = infos["extra.gamefile"][0]
        task_type = detect_task_type(gamefile)

        goal = extract_goal(observation)
        agent.reset(goal, task_type=task_type)

        print(f"\n{'=' * 60}")
        print(f"Game {game_idx + 1}/{alfred_env.num_games}  [{task_type}]")
        print(f"  Goal: {goal}")
        print(f"  Target object: '{agent.target_object}'")
        print(f"  Initial obs: {observation}")

        success = False
        num_steps = 0

        for step in range(max_steps):
            action = agent.plan_step(observation, admissible)

            state = agent._state_summary()
            print(f"  Step {step + 1}:")
            print(f"    Location: {state['current_location']}")
            print(f"    Holding: {state['holding']}")
            print(f"    Visited: {state['n_visited']}/{state['n_total']}")
            print(f"    Unvisited: {state['unvisited']}")
            print(f"    Todo: {agent.todo_list}")
            print(f"    Action: {action}")

            obs, rewards, dones, infos = env.step([action])
            observation = obs[0]
            admissible = list(infos["admissible_commands"][0])
            done = dones[0]
            won = infos["won"][0]
            num_steps = step + 1

            agent._update_state(observation, action, admissible)

            print(f"    Obs: {observation}")

            if done:
                success = won
                break

        status = "SUCCESS" if success else "FAIL"
        print(f"  Result: {status} in {num_steps} steps")

        results.append({
            "game_idx": game_idx,
            "gamefile": gamefile,
            "task_type": task_type,
            "goal": goal,
            "success": success,
            "steps": num_steps,
        })
        task_successes[task_type].append(success)

    env.close()

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    total_success = sum(r["success"] for r in results)
    total_games = len(results)
    print(f"\nOverall: {total_success}/{total_games}"
          f" ({100 * total_success / max(total_games, 1):.1f}%)")

    print("\nPer task type:")
    for tt in sorted(task_successes.keys()):
        successes = task_successes[tt]
        n_ok = sum(successes)
        n_all = len(successes)
        print(f"  {tt:40s}  {n_ok}/{n_all}  ({100 * n_ok / max(n_all, 1):.1f}%)")

    if output_file:
        output_data = {
            "config": {
                "agent": "todo_agent",
                "model": model,
                "api_url": api_url,
                "eval_split": eval_split,
                "max_steps": max_steps,
                "num_games": alfred_env.num_games,
            },
            "overall_success_rate": total_success / max(total_games, 1),
            "per_task_type": {
                tt: {"success": sum(s), "total": len(s),
                     "rate": sum(s) / max(len(s), 1)}
                for tt, s in task_successes.items()
            },
            "games": results,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Todo-List Planning Agent for ALFWorld"
    )
    parser.add_argument("config_file", help="Path to base_config.yaml")
    parser.add_argument("--num_games", type=int, default=-1,
                        help="Number of games to evaluate (-1 = all)")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--eval_split", default="eval_out_of_distribution",
                        choices=["eval_in_distribution",
                                 "eval_out_of_distribution"],
                        help="Evaluation split")
    parser.add_argument("--api_url",
                        default="https://tritonai-api.ucsd.edu/v1/chat/completions",
                        help="LLM API endpoint (OpenAI-compatible)")
    parser.add_argument("--model", default="api-llama-4-scout",
                        help="LLM model name")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    api_key = os.environ.get("TRITONAI_API_KEY", "")
    if not api_key:
        print("ERROR: Set TRITONAI_API_KEY environment variable.")
        sys.exit(1)

    assert os.path.exists(args.config_file), \
        f"Config file not found: {args.config_file}"
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    output_file = args.output
    if output_file is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"results_todo_{ts}.json"

    run_todo_agent(
        config=config,
        num_games=args.num_games,
        max_steps=args.max_steps,
        api_key=api_key,
        api_url=args.api_url,
        model=args.model,
        eval_split=args.eval_split,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
