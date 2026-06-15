from __future__ import annotations

import math
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from .config import (
    GRID_WIDTH, GRID_HEIGHT,
    HIVE_POS, INITIAL_NECTAR,
    INITIAL_NURSES, INITIAL_FORAGERS, INITIAL_SCOUTS,
    NUM_FLOWER_PATCHES,
    PATCH_MAX_NECTAR, PATCH_REGEN_RATE, PATCH_QUALITY_RANGE, MIN_PATCH_DISTANCE, PATCH_LIFETIME_NECTAR,
    WAGGLE_RECRUIT_MAX, WAGGLE_PROFITABILITY_SCALE, WAGGLE_ANGLE_NOISE,
    PHEROMONE_DECAY, PHEROMONE_DIFFUSION,
    USE_RL_FORAGERS,
)
from .agents.queen import QueenAgent
from .agents.nurse import NurseAgent
from .agents.forager import ForagerAgent, ForagerState
from .agents.rl_forager import RLForagerAgent
from .agents.scout import ScoutAgent
from .environment.hive import Hive
from .environment.flower_patch import FlowerPatch


class BeeColonyModel(Model):
    """
    Top-level model for the bee colony simulation.

    Owns the grid, scheduler, hive data, and flower patches.
    Coordinates the per-step sequence:
      1. Hatch mature eggs into new nurses.
      2. Regenerate flower patches.
      3. Collect data.
      4. Activate all scheduled agents.
    """

    def __init__(
        self,
        width: int = GRID_WIDTH,
        height: int = GRID_HEIGHT,
        initial_nurses: int = INITIAL_NURSES,
        initial_foragers: int = INITIAL_FORAGERS,
        initial_scouts: int = INITIAL_SCOUTS,
        num_patches: int = NUM_FLOWER_PATCHES,
        use_rl_foragers: bool = USE_RL_FORAGERS,
        seed: int | None = None,
    ):
        if width < 3 or height < 3:
            raise ValueError(f"Grid must be at least 3×3, got {width}×{height}.")

        super().__init__()
        if seed is not None:
            self.random.seed(seed)
        self.width = width
        self.height = height

        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)

        self.hive = Hive(HIVE_POS, INITIAL_NECTAR)
        self.flower_patches: list[FlowerPatch] = []
        self._patch_index: dict[tuple, FlowerPatch] = {}
        self.nurse_count: int = 0
        self.forager_count: int = 0
        self.scout_count: int = 0
        self.use_rl_foragers = use_rl_foragers
        self._forager_cls = RLForagerAgent if use_rl_foragers else ForagerAgent

        # first discovery per patch (used internally)
        self.use_pheromones: bool = True
        self.pheromones: np.ndarray = np.zeros((width, height), dtype=np.float32)

        self.patch_discoveries: list[dict] = []
        self._discovered_patches: set[tuple] = set()

        # first waggle dance per patch: [{found_step, finder, patch_pos, quality, recruits: [{forager_id, arrived_step}]}]
        self.dance_log: list[dict] = []
        self._danced_patches: set[tuple] = set()

        self._place_flower_patches(num_patches)
        self._spawn_initial_agents(initial_nurses, initial_foragers, initial_scouts)

        self.datacollector = DataCollector(
            model_reporters={
                "Nectar":   lambda m: round(m.hive.nectar, 1),
                "Nurses":   lambda m: m.nurse_count,
                "Foragers": lambda m: m.forager_count,
                "Scouts":   lambda m: m.scout_count,
                "Brood":    lambda m: m.hive.brood_count,
            }
        )

    # ── Per-step logic ───────────────────────────────────────────────────────

    def step(self) -> None:
        if (self.hive.nectar <= 0
                and self.forager_count == 0
                and self.nurse_count == 0):
            self.running = False
            return

        if self.use_pheromones:
            self._update_pheromones()

        new_nurses = self.hive.hatch(self.schedule.steps)
        for _ in range(new_nurses):
            nurse = NurseAgent(self)
            self.schedule.add(nurse)
            self.nurse_count += 1

        for patch in self.flower_patches:
            patch.step()

        self.datacollector.collect(self)
        self.schedule.step()

    def _update_pheromones(self) -> None:
        ph = self.pheromones
        w, h = ph.shape
        padded = np.pad(ph, 1, mode="constant")
        neighbor_avg = (
            padded[0:w,   0:h]   + padded[0:w,   1:h+1] + padded[0:w,   2:h+2] +
            padded[1:w+1, 0:h]   +                         padded[1:w+1, 2:h+2] +
            padded[2:w+2, 0:h]   + padded[2:w+2, 1:h+1] + padded[2:w+2, 2:h+2]
        ) / 8.0
        ph[:] = (ph * (1.0 - PHEROMONE_DIFFUSION) + PHEROMONE_DIFFUSION * neighbor_avg) * PHEROMONE_DECAY
        np.clip(ph, 0.0, 1.0, out=ph)

    # ── Waggle dance ─────────────────────────────────────────────────────────

    def perform_waggle_dance(self, patch, caller: str = "forager", found_step: int | None = None) -> None:
        """
        Recruit resting foragers at the hive to fly directly to `patch`.
        Logs the first dance per patch (forager or scout) in dance_log.
        found_step: step the agent landed on the patch (scouts pass this explicitly).
        """
        n = round(self._profitability_ratio(patch) * WAGGLE_RECRUIT_MAX)

        resting = [
            a for a in self.grid.get_cell_list_contents([self.hive.pos])
            if isinstance(a, ForagerAgent) and a.state == ForagerState.RESTING
        ]
        chosen = self.random.sample(resting, min(n, len(resting))) if n > 0 else []

        if patch.pos not in self._danced_patches:
            self._danced_patches.add(patch.pos)
            log_entry: dict = {
                "found_step": found_step if found_step is not None else self.schedule.steps,
                "finder":     caller,
                "patch_pos":  patch.pos,
                "quality":    round(patch.quality, 2),
                "recruits":   [],
            }
            self.dance_log.append(log_entry)
            for f in chosen:
                recruit_record = {"forager_id": f.unique_id, "arrived_step": None}
                log_entry["recruits"].append(recruit_record)
                f._scout_log_entry = recruit_record

        for f in chosen:
            f.target_patch = patch
            f.dance_target_pos = self._compute_dance_target(patch)
            f.state = ForagerState.FLYING_TO_PATCH
            f._rest_timer = 0

    def record_patch_discovery(self, patch, finder: str) -> None:
        """Log the first time any agent lands on a patch. Subsequent visits are ignored."""
        if patch.pos in self._discovered_patches:
            return
        self._discovered_patches.add(patch.pos)
        self.patch_discoveries.append({
            "step":    self.schedule.steps,
            "finder":  finder,
            "pos":     patch.pos,
            "quality": round(patch.quality, 2),
        })

    def _patch_profitability(self, patch) -> float:
        dx = patch.pos[0] - self.hive.pos[0]
        dy = patch.pos[1] - self.hive.pos[1]
        dist_sq = max(1, dx * dx + dy * dy)
        return patch.quality * patch.nectar / dist_sq

    def _profitability_ratio(self, patch) -> float:
        return min(1.0, self._patch_profitability(patch) / WAGGLE_PROFITABILITY_SCALE)

    def _compute_dance_target(self, patch) -> tuple[int, int]:
        hx, hy = self.hive.pos
        px, py = patch.pos
        dist = math.hypot(px - hx, py - hy)
        if dist < 1:
            return patch.pos
        angle = math.atan2(py - hy, px - hx)
        noisy_angle = angle + self.random.gauss(0, WAGGLE_ANGLE_NOISE)
        tx = round(hx + dist * math.cos(noisy_angle))
        ty = round(hy + dist * math.sin(noisy_angle))
        tx = max(0, min(self.width - 1, tx))
        ty = max(0, min(self.height - 1, ty))
        return (tx, ty)

    def promote_nurse_to_forager(self, nurse: NurseAgent) -> None:
        forager = self._forager_cls(self)
        self.grid.place_agent(forager, self.hive.pos)
        self.schedule.add(forager)
        self.schedule.remove(nurse)
        self.nurse_count -= 1
        self.forager_count += 1

    # ── Query helpers ────────────────────────────────────────────────────────

    def get_patch_at(self, pos: tuple) -> FlowerPatch | None:
        return self._patch_index.get(pos)

    def get_patch_near(self, pos: tuple, radius: int) -> FlowerPatch | None:
        """Return the closest non-depleted patch within Chebyshev `radius`, or None."""
        best: FlowerPatch | None = None
        best_dist = radius + 1
        px, py = pos
        for patch in self.flower_patches:
            if patch.is_depleted:
                continue
            dist = max(abs(patch.pos[0] - px), abs(patch.pos[1] - py))
            if dist <= radius and dist < best_dist:
                best_dist = dist
                best = patch
        return best

    # ── Initialisation ───────────────────────────────────────────────────────

    def _place_flower_patches(self, n: int) -> None:
        hx, hy = HIVE_POS
        x_range = (1, max(2, self.width - 1))
        y_range = (1, max(2, self.height - 1))
        placed, attempts = 0, 0
        while placed < n and attempts < n * 30:
            attempts += 1
            x = self.random.randrange(*x_range)
            y = self.random.randrange(*y_range)
            if abs(x - hx) + abs(y - hy) < MIN_PATCH_DISTANCE:
                continue
            pos = (x, y)
            if pos in self._patch_index:
                continue
            quality = self.random.uniform(*PATCH_QUALITY_RANGE)
            patch = FlowerPatch(pos, PATCH_MAX_NECTAR, quality, PATCH_REGEN_RATE, PATCH_LIFETIME_NECTAR)
            self.flower_patches.append(patch)
            self._patch_index[pos] = patch
            placed += 1

    def _spawn_initial_agents(self, n_nurses: int, n_foragers: int, n_scouts: int) -> None:
        self.schedule.add(QueenAgent(self))

        for _ in range(n_nurses):
            self.schedule.add(NurseAgent(self))
        self.nurse_count = n_nurses

        for _ in range(n_foragers):
            forager = self._forager_cls(self)
            self.grid.place_agent(forager, HIVE_POS)
            self.schedule.add(forager)
        self.forager_count = n_foragers

        for _ in range(n_scouts):
            scout = ScoutAgent(self)
            self.grid.place_agent(scout, HIVE_POS)
            self.schedule.add(scout)
        self.scout_count = n_scouts
