from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


TRAIL = 0
NECTAR = 1
HOME = 2

_ACTIONS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
)


@dataclass
class FlowerPatch:
    pos: tuple[int, int]
    max_nectar: float
    nectar: float
    quality: float
    regen_rate: float

    @property
    def is_depleted(self) -> bool:
        return self.nectar <= 0.05

    def collect(self, requested: float) -> float:
        raw = min(self.nectar, requested)
        self.nectar -= raw
        return raw * self.quality

    def step(self) -> None:
        self.nectar = min(self.max_nectar, self.nectar + self.regen_rate)


class BeeForagerEnv(gym.Env):
    """
    Small single-agent forager task.

    The agent learns the loop:
    find flower patch -> collect nectar -> return to hive -> repeat.

    Observation:
    0 dx to hive, normalized
    1 dy to hive, normalized
    2 dx to nearest non-empty patch, normalized
    3 dy to nearest non-empty patch, normalized
    4 nectar load ratio
    5 energy ratio
    6 local trail pheromone
    7 local nectar pheromone
    8 local home pheromone
    9 colony nectar deficit

    Action space:
    Discrete(9): stay plus 8 neighboring moves.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 25,
        n_patches: int = 4,
        max_steps: int = 300,
        load_capacity: float = 10.0,
        max_energy: float = 140.0,
        patch_max_nectar: float = 60.0,
        patch_regen_rate: float = 0.15,
        max_hive_nectar: float = 250.0,
        time_penalty: float = 0.01,
        collect_reward_scale: float = 0.5,
        deposit_reward_scale: float = 3.0,
        progress_reward_scale: float = 0.5,
        direction_reward_scale: float = 0.5,
        arrival_reward: float = 5.0,
        death_penalty: float = 10.0,
        seed: int | None = None,
    ):
        super().__init__()

        if grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        if n_patches < 1:
            raise ValueError("n_patches must be at least 1")

        self.grid_size = grid_size
        self.n_patches = n_patches
        self.max_steps = max_steps
        self.load_capacity = load_capacity
        self.max_energy = max_energy
        self.patch_max_nectar = patch_max_nectar
        self.patch_regen_rate = patch_regen_rate
        self.max_hive_nectar = max_hive_nectar
        self.time_penalty = time_penalty
        self.collect_reward_scale = collect_reward_scale
        self.deposit_reward_scale = deposit_reward_scale
        self.progress_reward_scale = progress_reward_scale
        self.direction_reward_scale = direction_reward_scale
        self.arrival_reward = arrival_reward
        self.death_penalty = death_penalty
        self.seed_value = seed
        self._seeded_once = False

        self.action_space = spaces.Discrete(len(_ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.hive_pos = (grid_size // 2, grid_size // 2)
        self.agent_pos = self.hive_pos
        self.energy = max_energy
        self.nectar_load = 0.0
        self.hive_nectar = 0.0
        self.steps = 0
        self.patches: list[FlowerPatch] = []
        self.pheromones = np.zeros((3, grid_size, grid_size), dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            actual_seed = seed
        elif not self._seeded_once:
            actual_seed = self.seed_value
        else:
            actual_seed = None
        super().reset(seed=actual_seed)
        if actual_seed is not None:
            self._seeded_once = True

        self.agent_pos = self.hive_pos
        self.energy = self.max_energy
        self.nectar_load = 0.0
        self.hive_nectar = 0.0
        self.steps = 0
        self.patches = self._make_patches()
        self._reset_pheromones()
        return self._get_obs(), {}

    def step(self, action: int):
        reward = -self.time_penalty
        self.steps += 1

        self._update_pheromones()
        for patch in self.patches:
            patch.step()

        old_pos = self.agent_pos
        old_target = self._current_target()
        old_distance = self._manhattan(old_pos, old_target)
        reward += self._direction_alignment(old_target, int(action)) * self.direction_reward_scale
        self._move(int(action))
        self.energy -= 1.0
        new_distance = self._manhattan(self.agent_pos, old_target)
        reward += (old_distance - new_distance) * self.progress_reward_scale

        if self.nectar_load > 0:
            x, y = old_pos
            self.pheromones[TRAIL, x, y] = min(
                1.0, self.pheromones[TRAIL, x, y] + 0.08
            )

        collected = self._try_collect()
        if collected > 0:
            reward += collected * self.collect_reward_scale
            if self.nectar_load >= self.load_capacity:
                reward += self.arrival_reward

        deposited = self._try_deposit()
        if deposited > 0:
            reward += deposited * self.deposit_reward_scale

        terminated = False
        truncated = self.steps >= self.max_steps

        if self.energy <= 0:
            reward -= self.death_penalty
            terminated = True

        info = {
            "collected": collected,
            "deposited": deposited,
            "hive_nectar": self.hive_nectar,
            "nectar_load": self.nectar_load,
            "energy": self.energy,
            "steps": self.steps,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        hx, hy = self.hive_pos
        grid[hy][hx] = "H"
        for patch in self.patches:
            px, py = patch.pos
            grid[py][px] = "F" if not patch.is_depleted else "f"
        ax, ay = self.agent_pos
        grid[ay][ax] = "B"
        rows = ["".join(row) for row in reversed(grid)]
        return "\n".join(rows)

    def heuristic_action(self) -> int:
        """Hand-written baseline: go to a patch when empty, go home when loaded."""
        return self._direction_action(self._current_target())

    def _move(self, action: int) -> None:
        dx, dy = _ACTIONS[max(0, min(len(_ACTIONS) - 1, action))]
        x, y = self.agent_pos
        self.agent_pos = (
            int(np.clip(x + dx, 0, self.grid_size - 1)),
            int(np.clip(y + dy, 0, self.grid_size - 1)),
        )

    def _current_target(self) -> tuple[int, int]:
        if self.nectar_load > 0:
            return self.hive_pos
        patch = self._nearest_patch()
        return patch.pos if patch is not None else self.hive_pos

    def _direction_action(self, target: tuple[int, int]) -> int:
        ax, ay = self.agent_pos
        tx, ty = target
        dx = int(np.sign(tx - ax))
        dy = int(np.sign(ty - ay))
        return _ACTIONS.index((dx, dy))

    def _direction_alignment(self, target: tuple[int, int], action: int) -> float:
        ax, ay = self.agent_pos
        tx, ty = target
        action_dx, action_dy = _ACTIONS[max(0, min(len(_ACTIONS) - 1, action))]
        old_distance = self._manhattan(self.agent_pos, target)
        new_pos = (
            int(np.clip(ax + action_dx, 0, self.grid_size - 1)),
            int(np.clip(ay + action_dy, 0, self.grid_size - 1)),
        )
        new_distance = self._manhattan(new_pos, target)

        if old_distance == 0:
            return 1.0 if (action_dx, action_dy) == (0, 0) else -0.5
        if new_distance < old_distance:
            return 1.0
        if new_distance > old_distance:
            return -1.0
        return -0.5

    def _try_collect(self) -> float:
        if self.nectar_load >= self.load_capacity:
            return 0.0

        patch = self._patch_at(self.agent_pos)
        if patch is None or patch.is_depleted:
            return 0.0

        free_capacity = self.load_capacity - self.nectar_load
        collected = patch.collect(free_capacity)
        self.nectar_load = min(self.load_capacity, self.nectar_load + collected)

        x, y = patch.pos
        self.pheromones[NECTAR, x, y] = min(1.0, self.pheromones[NECTAR, x, y] + 0.2)
        return collected

    def _try_deposit(self) -> float:
        if self.agent_pos != self.hive_pos or self.nectar_load <= 0:
            return 0.0

        deposited = self.nectar_load
        self.hive_nectar = min(self.max_hive_nectar, self.hive_nectar + deposited)
        self.nectar_load = 0.0
        self.energy = self.max_energy
        return deposited

    def _get_obs(self) -> np.ndarray:
        ax, ay = self.agent_pos
        hx, hy = self.hive_pos
        denom = max(1.0, self.grid_size - 1)

        patch = self._nearest_patch()
        if patch is None:
            px, py = ax, ay
        else:
            px, py = patch.pos

        local_pheromones = self.pheromones[:, ax, ay]
        colony_deficit = 1.0 - min(1.0, self.hive_nectar / self.max_hive_nectar)

        return np.array(
            [
                (hx - ax) / denom,
                (hy - ay) / denom,
                (px - ax) / denom,
                (py - ay) / denom,
                self.nectar_load / self.load_capacity,
                max(0.0, self.energy / self.max_energy),
                local_pheromones[TRAIL],
                local_pheromones[NECTAR],
                local_pheromones[HOME],
                colony_deficit,
            ],
            dtype=np.float32,
        )

    def _make_patches(self) -> list[FlowerPatch]:
        patches: list[FlowerPatch] = []
        min_dist_from_hive = max(3, self.grid_size // 6)
        attempts = 0

        while len(patches) < self.n_patches and attempts < self.n_patches * 100:
            attempts += 1
            x = int(self.np_random.integers(1, self.grid_size - 1))
            y = int(self.np_random.integers(1, self.grid_size - 1))
            pos = (x, y)
            if pos == self.hive_pos:
                continue
            if self._manhattan(pos, self.hive_pos) < min_dist_from_hive:
                continue
            if any(p.pos == pos for p in patches):
                continue

            quality = float(self.np_random.uniform(0.8, 1.3))
            patches.append(
                FlowerPatch(
                    pos=pos,
                    max_nectar=self.patch_max_nectar,
                    nectar=self.patch_max_nectar * 0.8,
                    quality=quality,
                    regen_rate=self.patch_regen_rate,
                )
            )

        if not patches:
            fallback = (self.grid_size - 3, self.grid_size - 3)
            patches.append(
                FlowerPatch(
                    pos=fallback,
                    max_nectar=self.patch_max_nectar,
                    nectar=self.patch_max_nectar * 0.8,
                    quality=1.0,
                    regen_rate=self.patch_regen_rate,
                )
            )
        return patches

    def _patch_at(self, pos: tuple[int, int]) -> FlowerPatch | None:
        for patch in self.patches:
            if patch.pos == pos:
                return patch
        return None

    def _nearest_patch(self) -> FlowerPatch | None:
        candidates = [p for p in self.patches if not p.is_depleted]
        if not candidates:
            return None
        return min(candidates, key=lambda p: self._manhattan(self.agent_pos, p.pos))

    def _reset_pheromones(self) -> None:
        self.pheromones.fill(0.0)
        hx, hy = self.hive_pos
        max_dist = max(1.0, 2 * (self.grid_size - 1))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dist = abs(x - hx) + abs(y - hy)
                self.pheromones[HOME, x, y] = max(0.0, 1.0 - dist / max_dist)

        self._mark_nectar_sources()

    def _update_pheromones(self) -> None:
        self.pheromones[TRAIL] = self._diffuse(self.pheromones[TRAIL], 0.02) * 0.995
        self.pheromones[NECTAR] = self._diffuse(self.pheromones[NECTAR], 0.03) * 0.985
        self._mark_nectar_sources()
        np.clip(self.pheromones, 0.0, 1.0, out=self.pheromones)

    def _mark_nectar_sources(self) -> None:
        for patch in self.patches:
            x, y = patch.pos
            strength = (patch.nectar / patch.max_nectar) * min(1.0, patch.quality / 1.3)
            self.pheromones[NECTAR, x, y] = max(self.pheromones[NECTAR, x, y], strength)

    @staticmethod
    def _diffuse(layer: np.ndarray, amount: float) -> np.ndarray:
        padded = np.pad(layer, 1, mode="edge")
        neighbor_avg = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 8.0
        return layer * (1.0 - amount) + neighbor_avg * amount

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
