from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.agents.rl_forager import _ACTIONS
from src.config import (
    FORAGER_ENERGY_COST_PER_STEP,
    FORAGER_LOAD_CAPACITY,
    FORAGER_MAX_ENERGY,
    MAX_NECTAR_STORES,
    NUM_FLOWER_PATCHES,
    SMELL_RADIUS,
    TRAIL_DEPOSIT_STRENGTH,
)
from src.model import BeeColonyModel


class BeeColonyForagerEnv(gym.Env):
    """
    Single PPO-controlled forager trained against the real colony primitives.

    This environment intentionally reuses BeeColonyModel's flower patches,
    hive, profitability calculation, and pheromone cellular automaton. It does
    not schedule colony agents; training focuses on one forager learning the
    collect-and-return loop with the same observation shape used by
    src.agents.rl_forager.RLForagerAgent.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_steps: int = 300,
        n_patches: int = NUM_FLOWER_PATCHES,
        use_pheromones: bool = True,
        seed: int | None = None,
        time_penalty: float = 0.01,
        collect_reward_scale: float = 0.5,
        deposit_reward_scale: float = 3.0,
        progress_reward_scale: float = 0.5,
        direction_reward_scale: float = 0.5,
        arrival_reward: float = 5.0,
        death_penalty: float = 10.0,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.n_patches = n_patches
        self.use_pheromones = use_pheromones
        self.seed_value = seed
        self.time_penalty = time_penalty
        self.collect_reward_scale = collect_reward_scale
        self.deposit_reward_scale = deposit_reward_scale
        self.progress_reward_scale = progress_reward_scale
        self.direction_reward_scale = direction_reward_scale
        self.arrival_reward = arrival_reward
        self.death_penalty = death_penalty

        self.action_space = spaces.Discrete(len(_ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.model: BeeColonyModel | None = None
        self.agent_pos: tuple[int, int] = (0, 0)
        self.energy = float(FORAGER_MAX_ENERGY)
        self.nectar_load = 0.0
        self.deposited_total = 0.0
        self.steps = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed if seed is not None else self.seed_value)
        actual_seed = seed if seed is not None else self.seed_value
        self.model = BeeColonyModel(
            initial_nurses=0,
            initial_foragers=0,
            initial_scouts=0,
            num_patches=self.n_patches,
            seed=actual_seed,
        )
        self.model.use_pheromones = self.use_pheromones
        self.agent_pos = self.model.hive.pos
        self.energy = float(FORAGER_MAX_ENERGY)
        self.nectar_load = 0.0
        self.deposited_total = 0.0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: int):
        model = self._model()
        reward = -self.time_penalty
        self.steps += 1

        if model.use_pheromones:
            model._update_pheromones()
        for patch in model.flower_patches:
            patch.step()

        old_pos = self.agent_pos
        old_target = self._current_target()
        old_distance = self._manhattan(old_pos, old_target)
        reward += self._direction_alignment(old_target, int(action)) * self.direction_reward_scale

        self._move(int(action))
        self.energy -= FORAGER_ENERGY_COST_PER_STEP

        new_distance = self._manhattan(self.agent_pos, old_target)
        reward += (old_distance - new_distance) * self.progress_reward_scale

        if self.nectar_load > 0 and model.use_pheromones:
            self._deposit_trail(old_pos)

        collected = self._try_collect()
        if collected > 0:
            reward += collected * self.collect_reward_scale
            if self.nectar_load >= FORAGER_LOAD_CAPACITY:
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
            "deposited_total": self.deposited_total,
            "nectar_load": self.nectar_load,
            "energy": self.energy,
            "steps": self.steps,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def heuristic_action(self) -> int:
        return self._direction_action(self._current_target())

    def render(self):
        model = self._model()
        grid = [["." for _ in range(model.width)] for _ in range(model.height)]
        hx, hy = model.hive.pos
        grid[hy][hx] = "H"
        for patch in model.flower_patches:
            px, py = patch.pos
            grid[py][px] = "F" if not patch.is_depleted else "f"
        ax, ay = self.agent_pos
        grid[ay][ax] = "B"
        return "\n".join("".join(row) for row in reversed(grid))

    def _move(self, action: int) -> None:
        model = self._model()
        dx, dy = _ACTIONS[max(0, min(len(_ACTIONS) - 1, action))]
        x, y = self.agent_pos
        self.agent_pos = (
            int(np.clip(x + dx, 0, model.width - 1)),
            int(np.clip(y + dy, 0, model.height - 1)),
        )

    def _try_collect(self) -> float:
        if self.nectar_load >= FORAGER_LOAD_CAPACITY:
            return 0.0
        patch = self._model().get_patch_at(self.agent_pos)
        if patch is None or patch.is_depleted:
            return 0.0
        collected = patch.collect(FORAGER_LOAD_CAPACITY - self.nectar_load)
        self.nectar_load = min(FORAGER_LOAD_CAPACITY, self.nectar_load + collected)
        self._deposit_trail(patch.pos)
        return collected

    def _try_deposit(self) -> float:
        model = self._model()
        if self.agent_pos != model.hive.pos or self.nectar_load <= 0:
            return 0.0
        deposited = self.nectar_load
        model.hive.deposit(deposited)
        self.deposited_total += deposited
        self.nectar_load = 0.0
        self.energy = float(FORAGER_MAX_ENERGY)
        return deposited

    def _get_obs(self) -> np.ndarray:
        model = self._model()
        ax, ay = self.agent_pos
        hx, hy = model.hive.pos
        denom = max(1.0, max(model.width, model.height) - 1)
        patch = self._nearest_patch()
        px, py = patch.pos if patch is not None else self.agent_pos
        local_trail = float(model.pheromones[ax, ay]) if model.use_pheromones else 0.0
        colony_deficit = 1.0 - min(1.0, model.hive.nectar / MAX_NECTAR_STORES)

        return np.array(
            [
                (hx - ax) / denom,
                (hy - ay) / denom,
                (px - ax) / denom,
                (py - ay) / denom,
                self.nectar_load / FORAGER_LOAD_CAPACITY,
                max(0.0, self.energy / FORAGER_MAX_ENERGY),
                local_trail,
                self._local_nectar_signal(ax, ay),
                self._home_signal(ax, ay),
                colony_deficit,
            ],
            dtype=np.float32,
        )

    def _current_target(self) -> tuple[int, int]:
        if self.nectar_load > 0:
            return self._model().hive.pos
        patch = self._nearest_patch()
        return patch.pos if patch is not None else self._model().hive.pos

    def _nearest_patch(self):
        candidates = [p for p in self._model().flower_patches if not p.is_depleted]
        if not candidates:
            return None
        return min(candidates, key=lambda p: self._manhattan(self.agent_pos, p.pos))

    def _direction_action(self, target: tuple[int, int]) -> int:
        ax, ay = self.agent_pos
        tx, ty = target
        return _ACTIONS.index((int(np.sign(tx - ax)), int(np.sign(ty - ay))))

    def _direction_alignment(self, target: tuple[int, int], action: int) -> float:
        model = self._model()
        ax, ay = self.agent_pos
        action_dx, action_dy = _ACTIONS[max(0, min(len(_ACTIONS) - 1, action))]
        old_distance = self._manhattan(self.agent_pos, target)
        new_pos = (
            int(np.clip(ax + action_dx, 0, model.width - 1)),
            int(np.clip(ay + action_dy, 0, model.height - 1)),
        )
        new_distance = self._manhattan(new_pos, target)

        if old_distance == 0:
            return 1.0 if (action_dx, action_dy) == (0, 0) else -0.5
        if new_distance < old_distance:
            return 1.0
        if new_distance > old_distance:
            return -1.0
        return -0.5

    def _deposit_trail(self, pos: tuple[int, int]) -> None:
        model = self._model()
        if not model.use_pheromones:
            return
        patch = self._nearest_patch()
        if patch is None:
            return
        x, y = pos
        deposit = model._profitability_ratio(patch) * TRAIL_DEPOSIT_STRENGTH
        model.pheromones[x, y] = min(1.0, model.pheromones[x, y] + deposit)

    def _local_nectar_signal(self, x: int, y: int) -> float:
        signal = 0.0
        for patch in self._model().flower_patches:
            if patch.exhausted:
                continue
            dist = max(abs(patch.pos[0] - x), abs(patch.pos[1] - y))
            if dist <= SMELL_RADIUS:
                fullness = patch.nectar / patch.max_nectar if patch.max_nectar else 0.0
                quality = min(1.0, patch.quality / 1.5)
                signal = max(signal, fullness * quality * (1.0 - dist / (SMELL_RADIUS + 1)))
        return float(np.clip(signal, 0.0, 1.0))

    def _home_signal(self, x: int, y: int) -> float:
        model = self._model()
        hx, hy = model.hive.pos
        max_dist = max(1.0, (model.width - 1) + (model.height - 1))
        dist = abs(x - hx) + abs(y - hy)
        return float(max(0.0, 1.0 - dist / max_dist))

    def _model(self) -> BeeColonyModel:
        if self.model is None:
            raise RuntimeError("Environment has not been reset")
        return self.model

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
