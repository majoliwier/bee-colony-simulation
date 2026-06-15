from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    FORAGER_ENERGY_COST_PER_STEP,
    FORAGER_LOAD_CAPACITY,
    FORAGER_MAX_ENERGY,
    MAX_NECTAR_STORES,
    RL_FORAGER_MODEL_PATH,
    RL_FORAGER_REST_DURATION,
    SMELL_RADIUS,
    TRAIL_DEPOSIT_STRENGTH,
)
from .forager import ForagerAgent, ForagerState


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


class RLForagerAgent(ForagerAgent):
    """Forager controlled by a PPO policy trained in rl_forager_mvp.

    The agent keeps the public fields used by the visualisation and waggle
    dance logic, but replaces the finite-state movement policy with a learned
    action over the 8-neighbourhood plus "stay".
    """

    _policy: Any | None = None
    _policy_load_failed: bool = False

    def step(self) -> None:
        if self.state == ForagerState.RESTING:
            self._rest_timer += 1
            self.energy = float(FORAGER_MAX_ENERGY)
            if self._rest_timer < RL_FORAGER_REST_DURATION:
                return
            self._rest_timer = 0

        self.energy -= FORAGER_ENERGY_COST_PER_STEP
        if self.energy <= 0:
            self._die()
            return

        old_pos = self.pos
        action = self._choose_action()
        self._move_by_action(action)

        if self.nectar_load > 0 and old_pos is not None:
            self._deposit_trail_at(old_pos)

        collected = self._try_collect()
        deposited = self._try_deposit()
        self._update_visual_state(collected, deposited)

    def _choose_action(self) -> int:
        obs = self._get_obs()
        policy = self._get_policy()
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
            if self._action_reduces_distance(action):
                return action
        return self._heuristic_action()

    @classmethod
    def _get_policy(cls):
        if cls._policy is not None or cls._policy_load_failed:
            return cls._policy

        try:
            from stable_baselines3 import PPO
        except ImportError:
            cls._policy_load_failed = True
            return None

        model_path = Path(RL_FORAGER_MODEL_PATH)
        if not model_path.exists():
            cls._policy_load_failed = True
            return None

        cls._policy = PPO.load(str(model_path))
        return cls._policy

    def _move_by_action(self, action: int) -> None:
        dx, dy = _ACTIONS[max(0, min(len(_ACTIONS) - 1, int(action)))]
        x, y = self.pos
        new_pos = (
            int(np.clip(x + dx, 0, self.model.width - 1)),
            int(np.clip(y + dy, 0, self.model.height - 1)),
        )
        self.model.grid.move_agent(self, new_pos)

    def _try_collect(self) -> float:
        if self.nectar_load >= FORAGER_LOAD_CAPACITY:
            return 0.0

        patch = self.model.get_patch_at(self.pos)
        if patch is None or patch.is_depleted:
            nearby = self.model.get_patch_near(self.pos, SMELL_RADIUS)
            if nearby is not None and not nearby.is_depleted:
                self.target_patch = nearby
            return 0.0

        collected = patch.collect(FORAGER_LOAD_CAPACITY - self.nectar_load)
        if collected <= 0:
            return 0.0

        self.nectar_load = min(FORAGER_LOAD_CAPACITY, self.nectar_load + collected)
        self.target_patch = patch
        self._found_at_step = self.model.schedule.steps
        self.model.record_patch_discovery(patch, "rl_forager")
        self._deposit_trail_at(patch.pos)
        self._broadcast_patch_marker()
        return collected

    def _try_deposit(self) -> float:
        if self.pos != self.model.hive.pos or self.nectar_load <= 0:
            return 0.0

        deposited = self.nectar_load
        self.model.hive.deposit(deposited)
        self.nectar_load = 0.0
        self.energy = float(FORAGER_MAX_ENERGY)
        if self.target_patch is not None and not self.target_patch.exhausted:
            self.model.perform_waggle_dance(
                self.target_patch,
                caller="rl_forager",
                found_step=self._found_at_step,
            )
        self.target_patch = None
        self.dance_target_pos = None
        self._scout_log_entry = None
        return deposited

    def _update_visual_state(self, collected: float, deposited: float) -> None:
        if deposited > 0:
            self.state = ForagerState.RESTING
        elif collected > 0:
            self.state = ForagerState.COLLECTING
        elif self.nectar_load > 0:
            self.state = ForagerState.RETURNING
        elif self.target_patch is not None:
            self.state = ForagerState.FLYING_TO_PATCH
        else:
            self.state = ForagerState.SCOUTING

    def _get_obs(self) -> np.ndarray:
        ax, ay = self.pos
        hx, hy = self.model.hive.pos
        denom = max(1.0, max(self.model.width, self.model.height) - 1)

        patch = self._observation_patch()
        if patch is None:
            px, py = ax, ay
        else:
            px, py = patch.pos

        local_trail = float(self.model.pheromones[ax, ay]) if self.model.use_pheromones else 0.0
        local_nectar = self._local_nectar_signal(ax, ay)
        local_home = self._home_signal(ax, ay)
        colony_deficit = 1.0 - min(1.0, self.model.hive.nectar / MAX_NECTAR_STORES)

        return np.array(
            [
                (hx - ax) / denom,
                (hy - ay) / denom,
                (px - ax) / denom,
                (py - ay) / denom,
                self.nectar_load / FORAGER_LOAD_CAPACITY,
                max(0.0, self.energy / FORAGER_MAX_ENERGY),
                local_trail,
                local_nectar,
                local_home,
                colony_deficit,
            ],
            dtype=np.float32,
        )

    def _observation_patch(self):
        if self.target_patch is not None and not self.target_patch.is_depleted:
            return self.target_patch
        return self._nearest_patch()

    def _nearest_patch(self):
        candidates = [p for p in self.model.flower_patches if not p.is_depleted]
        if not candidates:
            return None
        return min(candidates, key=lambda p: self._manhattan(self.pos, p.pos))

    def _heuristic_action(self) -> int:
        if self.nectar_load > 0:
            target = self.model.hive.pos
        else:
            patch = self._observation_patch()
            target = patch.pos if patch is not None else self.model.hive.pos

        ax, ay = self.pos
        tx, ty = target
        dx = int(np.sign(tx - ax))
        dy = int(np.sign(ty - ay))
        return _ACTIONS.index((dx, dy))

    def _action_reduces_distance(self, action: int) -> bool:
        if self.nectar_load > 0:
            target = self.model.hive.pos
        else:
            patch = self._observation_patch()
            target = patch.pos if patch is not None else self.model.hive.pos

        old_distance = self._manhattan(self.pos, target)
        if old_distance == 0:
            return action == 0

        dx, dy = _ACTIONS[max(0, min(len(_ACTIONS) - 1, int(action)))]
        x, y = self.pos
        new_pos = (
            int(np.clip(x + dx, 0, self.model.width - 1)),
            int(np.clip(y + dy, 0, self.model.height - 1)),
        )
        return self._manhattan(new_pos, target) < old_distance

    def _deposit_trail_at(self, pos: tuple[int, int]) -> None:
        if self.target_patch is None or not self.model.use_pheromones:
            return
        deposit = self.model._profitability_ratio(self.target_patch) * TRAIL_DEPOSIT_STRENGTH
        x, y = pos
        self.model.pheromones[x, y] = min(1.0, self.model.pheromones[x, y] + deposit)

    def _local_nectar_signal(self, x: int, y: int) -> float:
        signal = 0.0
        for patch in self.model.flower_patches:
            if patch.exhausted:
                continue
            dist = max(abs(patch.pos[0] - x), abs(patch.pos[1] - y))
            if dist <= SMELL_RADIUS:
                fullness = patch.nectar / patch.max_nectar if patch.max_nectar else 0.0
                quality = min(1.0, patch.quality / 1.5)
                signal = max(signal, fullness * quality * (1.0 - dist / (SMELL_RADIUS + 1)))
        return float(np.clip(signal, 0.0, 1.0))

    def _home_signal(self, x: int, y: int) -> float:
        hx, hy = self.model.hive.pos
        max_dist = max(1.0, (self.model.width - 1) + (self.model.height - 1))
        dist = abs(x - hx) + abs(y - hy)
        return float(max(0.0, 1.0 - dist / max_dist))

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
