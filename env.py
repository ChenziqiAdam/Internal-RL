"""
Gridworld Pinpad Environment
Per paper Appendix A.1:
- 7×7 grid, 8 colored cells, 4 walls, 4 cardinal actions
- Agent visits colored cells in task-specified order
- Episode ends on: success, wrong visit, or T=100 steps
- Observation: one-hot of cell contents + agent position (7²×13 dims)
- Sparse reward: 1 on full task completion, 0 otherwise
"""

import numpy as np
from typing import List, Tuple, Optional

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Cell types
EMPTY = 0
WALL = 1
COLORS = list(range(2, 10))  # 8 colors: 2..9
NUM_COLORS = 8

# Observation channels: empty(1) + wall(1) + 8 colors + agent(1) + goal_progress(1) = 13
# Actually per paper: "one-hot encoding of cell contents + agent position"
# cell contents: empty(1) + wall(1) + 8 colors = 10 channels
# agent position: 1 channel
# We'll use: 11 channels per cell → obs shape (7, 7, 11) → flattened 7*7*11 = 539
# Wait - paper says 7²×(8+4+1) = 49×13 = 637
# 8 colors + 4 directions(?) + 1 agent = 13. Let's match paper exactly.
# Re-reading: "(7²×(8+4+1) dims)" - we'll interpret as 8 cell-type one-hots + 4 wall indicators + 1 agent
# More likely: 8 color channels + 1 wall + 1 empty + 1 agent = 11, but paper says 13.
# Best interpretation: 8 colors + 1 wall + 1 empty + 1 agent + 1 visited + 1 goal = 13
# We'll use: cell_type (one-hot 10: empty/wall/8colors) + is_agent (1) + progress_step (1) + is_goal_cell (1)
# = 13 channels per cell

CELL_CHANNELS = 13
GRID_SIZE = 7
OBS_DIM = GRID_SIZE * GRID_SIZE * CELL_CHANNELS  # 637
NUM_ACTIONS = 4
NUM_WALLS = 4
MAX_STEPS = 100

# 16 Pretraining tasks (Table A1) — sequences of color indices 0..7
# These are short sequences visiting a subset of colors in order
PRETRAIN_TASKS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 0],
    [0, 1, 2],
    [2, 3, 4],
    [4, 5, 6],
    [6, 7, 0],
    [0, 2, 4],
    [1, 3, 5],
    [2, 4, 6],
    [3, 5, 7],
]

# Post-training task: 0-1-2-3-4-5-6-7-0-1-2-3 (12 steps, unseen combination)
POSTRAIN_TASK = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]


class GridworldPinpad:
    """
    Gridworld pinpad environment.
    Task = ordered sequence of color indices to visit.
    Each episode samples a random layout (cell positions, walls, agent start).
    """

    def __init__(self, task: List[int], seed: Optional[int] = None):
        self.task = task  # list of color indices (0..7)
        self.rng = np.random.default_rng(seed)
        self.grid = None
        self.agent_pos = None
        self.step_count = 0
        self.task_step = 0  # how many colors in task have been visited
        self.color_positions = {}  # color_idx -> (r, c)
        self.done = False

    def reset(self) -> np.ndarray:
        self._sample_layout()
        self.step_count = 0
        self.task_step = 0
        self.done = False
        return self._get_obs()

    def _sample_layout(self):
        """Sample random positions for 8 colored cells, 4 walls, and agent start."""
        all_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        indices = self.rng.permutation(len(all_cells))

        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

        # Place 8 colored cells
        self.color_positions = {}
        for i, color_idx in enumerate(range(NUM_COLORS)):
            r, c = all_cells[indices[i]]
            self.grid[r, c] = COLORS[color_idx]
            self.color_positions[color_idx] = (r, c)

        # Place 4 walls
        for i in range(NUM_COLORS, NUM_COLORS + NUM_WALLS):
            r, c = all_cells[indices[i]]
            self.grid[r, c] = WALL

        # Agent start (empty cell)
        agent_idx = NUM_COLORS + NUM_WALLS
        r, c = all_cells[indices[agent_idx]]
        self.agent_pos = (r, c)
        # Agent doesn't overwrite grid cell (agent position tracked separately)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert not self.done, "Episode done, call reset()"
        assert action in ACTION_DELTAS

        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # Clip to grid bounds; wall = stay in place
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and self.grid[nr, nc] != WALL:
            self.agent_pos = (nr, nc)

        reward = 0.0
        self.step_count += 1

        # Check if agent is on a colored cell
        cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
        if cell in COLORS:
            color_idx = COLORS.index(cell)
            expected_color = self.task[self.task_step]

            if color_idx == expected_color:
                self.task_step += 1
                if self.task_step == len(self.task):
                    reward = 1.0
                    self.done = True
            else:
                # Wrong color visited
                self.done = True

        if self.step_count >= MAX_STEPS:
            self.done = True

        return self._get_obs(), reward, self.done, {"task_step": self.task_step}

    def _get_obs(self) -> np.ndarray:
        """
        Build observation: GRID_SIZE × GRID_SIZE × CELL_CHANNELS, flattened.
        Channels per cell:
          0: empty
          1: wall
          2..9: color 0..7
          10: agent present
          11: is current goal cell (next color to visit)
          12: is completed goal (already visited in sequence)
        """
        obs = np.zeros((GRID_SIZE, GRID_SIZE, CELL_CHANNELS), dtype=np.float32)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = self.grid[r, c]
                if cell == EMPTY:
                    obs[r, c, 0] = 1.0
                elif cell == WALL:
                    obs[r, c, 1] = 1.0
                else:
                    color_idx = COLORS.index(cell)
                    obs[r, c, 2 + color_idx] = 1.0

        # Agent position
        ar, ac = self.agent_pos
        obs[ar, ac, 10] = 1.0

        # Current goal cell
        if self.task_step < len(self.task):
            goal_color = self.task[self.task_step]
            gr, gc = self.color_positions[goal_color]
            obs[gr, gc, 11] = 1.0

        # Completed subgoals
        for i in range(self.task_step):
            past_color = self.task[i]
            pr, pc = self.color_positions[past_color]
            obs[pr, pc, 12] = 1.0

        return obs.reshape(-1)  # flatten to OBS_DIM

    def get_subgoal_label(self) -> int:
        """Return current subgoal index (task_step) for probing/supervision."""
        return self.task_step

    def get_color_positions(self) -> dict:
        return dict(self.color_positions)


def make_env(task: List[int], seed: Optional[int] = None) -> GridworldPinpad:
    return GridworldPinpad(task, seed=seed)
