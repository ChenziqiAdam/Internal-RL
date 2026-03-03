"""
Phase 2: Expert Data Generation
- BFS shortest-path solver per task/layout
- Generate optimal trajectories (ε=0)
- Store as (obs_seq, action_seq) with subgoal labels
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
import pickle
import os

from env import (
    GridworldPinpad, PRETRAIN_TASKS, POSTRAIN_TASK,
    GRID_SIZE, WALL, COLORS, ACTION_DELTAS, UP, DOWN, LEFT, RIGHT,
    OBS_DIM, MAX_STEPS
)


def bfs_shortest_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[int]]:
    """
    BFS to find shortest action sequence from start to goal.
    Returns list of actions, or None if unreachable.
    """
    if start == goal:
        return []

    visited = {start}
    queue = deque([(start, [])])

    while queue:
        pos, path = queue.popleft()
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            npos = (nr, nc)
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] != WALL and npos not in visited:
                new_path = path + [action]
                if npos == goal:
                    return new_path
                visited.add(npos)
                queue.append((npos, new_path))

    return None  # unreachable


def solve_episode(env: GridworldPinpad) -> Optional[Tuple[List[np.ndarray], List[int], List[int]]]:
    """
    Solve one episode using BFS through the task sequence.
    Returns (obs_list, action_list, subgoal_labels) or None if unsolvable.
    """
    obs = env.reset()
    obs_list = [obs.copy()]
    action_list = []
    subgoal_labels = []  # subgoal index at each step

    for subgoal_idx, color_idx in enumerate(env.task):
        goal_pos = env.color_positions[color_idx]
        path = bfs_shortest_path(env.grid, env.agent_pos, goal_pos)

        if path is None:
            return None  # layout is unsolvable

        for action in path:
            subgoal_labels.append(subgoal_idx)
            obs, reward, done, info = env.step(action)
            action_list.append(action)
            obs_list.append(obs.copy())

            if done and reward == 0:
                return None  # hit wrong cell somehow (shouldn't happen with BFS)
            if done:
                break

        if env.task_step != subgoal_idx + 1:
            # Didn't advance — something wrong
            return None

    if len(action_list) > MAX_STEPS:
        return None

    return obs_list, action_list, subgoal_labels


def generate_dataset(
    tasks: List[List[int]],
    num_episodes_per_task: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Generate expert trajectories for given tasks.
    Returns dict with keys: obs_seqs, action_seqs, subgoal_seqs, task_ids
    """
    rng = np.random.default_rng(seed)
    obs_seqs = []
    action_seqs = []
    subgoal_seqs = []
    task_ids = []

    for task_idx, task in enumerate(tasks):
        count = 0
        attempts = 0
        while count < num_episodes_per_task:
            ep_seed = int(rng.integers(0, 2**31))
            env = GridworldPinpad(task, seed=ep_seed)
            result = solve_episode(env)
            attempts += 1

            if result is not None:
                obs_list, action_list, subgoal_list = result
                obs_seqs.append(np.array(obs_list, dtype=np.float32))
                action_seqs.append(np.array(action_list, dtype=np.int32))
                subgoal_seqs.append(np.array(subgoal_list, dtype=np.int32))
                task_ids.append(task_idx)
                count += 1

            if attempts > num_episodes_per_task * 10:
                print(f"Warning: task {task_idx} only got {count} episodes after {attempts} attempts")
                break

        print(f"Task {task_idx} ({task}): {count} episodes generated")

    return {
        "obs_seqs": obs_seqs,
        "action_seqs": action_seqs,
        "subgoal_seqs": subgoal_seqs,
        "task_ids": task_ids,
        "tasks": tasks,
    }


def save_dataset(dataset: Dict, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset['task_ids'])} episodes to {path}")


def load_dataset(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["pretrain", "posttrain"], default="pretrain")
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    if args.split == "pretrain":
        tasks = PRETRAIN_TASKS
        out = os.path.join(args.out_dir, "pretrain.pkl")
    else:
        tasks = [POSTRAIN_TASK]
        out = os.path.join(args.out_dir, "posttrain.pkl")

    dataset = generate_dataset(tasks, num_episodes_per_task=args.n_episodes, seed=args.seed)
    save_dataset(dataset, out)
