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


def save_dataset_mmap(dataset: Dict, out_dir: str):
    """
    Save dataset as memory-mapped flat arrays + index file.
    Files created:
      obs_flat.npy   — shape (total_steps+N, obs_dim) float32
      acts_flat.npy  — shape (total_steps,) int32
      subgoals_flat.npy — shape (total_steps,) int32
      index.npy      — shape (N, 2) int64, start/end offsets per episode
      meta.npy       — pickled task_ids and tasks (stored as object array)
    """
    os.makedirs(out_dir, exist_ok=True)
    obs_seqs = dataset["obs_seqs"]
    act_seqs = dataset["action_seqs"]
    sg_seqs = dataset["subgoal_seqs"]
    N = len(obs_seqs)

    # Build index: (start_step, end_step) for each episode (step = action index)
    lengths = [len(a) for a in act_seqs]
    index = np.zeros((N, 2), dtype=np.int64)
    offset = 0
    for i, L in enumerate(lengths):
        index[i] = [offset, offset + L]
        offset += L
    total_steps = offset

    obs_dim = obs_seqs[0].shape[1]

    # Allocate and fill flat arrays
    obs_flat = np.empty((total_steps + N, obs_dim), dtype=np.float32)
    acts_flat = np.empty(total_steps, dtype=np.int32)
    sg_flat = np.empty(total_steps, dtype=np.int32)

    obs_offset = 0
    for i, (obs, act, sg) in enumerate(zip(obs_seqs, act_seqs, sg_seqs)):
        L = len(act)
        # obs has T+1 entries; acts/sg have T entries
        obs_flat[obs_offset: obs_offset + L + 1] = obs
        acts_flat[index[i, 0]: index[i, 1]] = act
        sg_flat[index[i, 0]: index[i, 1]] = sg
        obs_offset += L + 1

    # obs_index: (start_obs, end_obs) per episode
    obs_index = np.zeros((N, 2), dtype=np.int64)
    obs_off = 0
    for i, L in enumerate(lengths):
        obs_index[i] = [obs_off, obs_off + L + 1]
        obs_off += L + 1

    np.save(os.path.join(out_dir, "obs_flat.npy"), obs_flat)
    np.save(os.path.join(out_dir, "acts_flat.npy"), acts_flat)
    np.save(os.path.join(out_dir, "subgoals_flat.npy"), sg_flat)
    np.save(os.path.join(out_dir, "index.npy"), index)           # action offsets
    np.save(os.path.join(out_dir, "obs_index.npy"), obs_index)   # obs offsets
    meta = {"task_ids": dataset["task_ids"], "tasks": dataset["tasks"]}
    np.save(os.path.join(out_dir, "meta.npy"), meta, allow_pickle=True)
    print(f"Saved mmap dataset ({N} episodes, {total_steps} steps) to {out_dir}/")


def load_dataset_mmap(out_dir: str) -> Dict:
    """
    Load mmap dataset. Arrays are memory-mapped (read-only); only accessed slices
    are paged into RAM.
    """
    obs_flat = np.load(os.path.join(out_dir, "obs_flat.npy"), mmap_mode="r")
    acts_flat = np.load(os.path.join(out_dir, "acts_flat.npy"), mmap_mode="r")
    sg_flat = np.load(os.path.join(out_dir, "subgoals_flat.npy"), mmap_mode="r")
    index = np.load(os.path.join(out_dir, "index.npy"))
    obs_index = np.load(os.path.join(out_dir, "obs_index.npy"))
    meta = np.load(os.path.join(out_dir, "meta.npy"), allow_pickle=True).item()
    return {
        "obs_flat": obs_flat,
        "acts_flat": acts_flat,
        "subgoals_flat": sg_flat,
        "index": index,
        "obs_index": obs_index,
        "task_ids": meta["task_ids"],
        "tasks": meta["tasks"],
        "_is_mmap": True,
    }


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
        pkl_out = os.path.join(args.out_dir, "pretrain.pkl")
        mmap_out = os.path.join(args.out_dir, "pretrain")
    else:
        tasks = [POSTRAIN_TASK]
        pkl_out = os.path.join(args.out_dir, "posttrain.pkl")
        mmap_out = os.path.join(args.out_dir, "posttrain")

    dataset = generate_dataset(tasks, num_episodes_per_task=args.n_episodes, seed=args.seed)
    save_dataset(dataset, pkl_out)
    save_dataset_mmap(dataset, mmap_out)
