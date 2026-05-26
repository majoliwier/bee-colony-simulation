from __future__ import annotations

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from .config import (
    GRID_WIDTH, GRID_HEIGHT,
    HIVE_POS, INITIAL_NECTAR,
    INITIAL_NURSES, INITIAL_FORAGERS, INITIAL_SCOUTS,
    NUM_FLOWER_PATCHES,
    PATCH_MAX_NECTAR, PATCH_REGEN_RATE, PATCH_QUALITY_RANGE, MIN_PATCH_DISTANCE,
    WAGGLE_RECRUIT_MAX, WAGGLE_PROFITABILITY_SCALE,
)
from .agents.queen import QueenAgent
from .agents.nurse import NurseAgent
from .agents.forager import ForagerAgent, ForagerState
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
    ):
        if width < 3 or height < 3:
            raise ValueError(f"Grid must be at least 3×3, got {width}×{height}.")

        super().__init__()
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
        new_nurses = self.hive.hatch(self.schedule.steps)
        for _ in range(new_nurses):
            nurse = NurseAgent(self)
            self.schedule.add(nurse)
            self.nurse_count += 1

        for patch in self.flower_patches:
            patch.step()

        self.datacollector.collect(self)
        self.schedule.step()

    # ── Waggle dance ─────────────────────────────────────────────────────────

    def perform_waggle_dance(self, patch) -> None:
        """
        Recruit resting foragers at the hive to fly directly to `patch`.
        Called by returning foragers and scouts on hive arrival.
        """
        prof = self._patch_profitability(patch)
        n = min(
            WAGGLE_RECRUIT_MAX,
            round(min(1.0, prof / WAGGLE_PROFITABILITY_SCALE) * WAGGLE_RECRUIT_MAX),
        )
        if n <= 0:
            return
        resting = [
            a for a in self.grid.get_cell_list_contents([self.hive.pos])
            if isinstance(a, ForagerAgent) and a.state == ForagerState.RESTING
        ]
        chosen = self.random.sample(resting, min(n, len(resting)))
        for f in chosen:
            f.target_patch = patch
            f.state = ForagerState.FLYING_TO_PATCH
            f._rest_timer = 0

    def _patch_profitability(self, patch) -> float:
        dx = patch.pos[0] - self.hive.pos[0]
        dy = patch.pos[1] - self.hive.pos[1]
        dist_sq = max(1, dx * dx + dy * dy)
        return patch.quality * patch.nectar / dist_sq

    # ── Query helpers ────────────────────────────────────────────────────────

    def get_patch_at(self, pos: tuple) -> FlowerPatch | None:
        return self._patch_index.get(pos)

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
            patch = FlowerPatch(pos, PATCH_MAX_NECTAR, quality, PATCH_REGEN_RATE)
            self.flower_patches.append(patch)
            self._patch_index[pos] = patch
            placed += 1

    def _spawn_initial_agents(self, n_nurses: int, n_foragers: int, n_scouts: int) -> None:
        self.schedule.add(QueenAgent(self))

        for _ in range(n_nurses):
            self.schedule.add(NurseAgent(self))
        self.nurse_count = n_nurses

        for _ in range(n_foragers):
            forager = ForagerAgent(self)
            self.grid.place_agent(forager, HIVE_POS)
            self.schedule.add(forager)
        self.forager_count = n_foragers

        for _ in range(n_scouts):
            scout = ScoutAgent(self)
            self.grid.place_agent(scout, HIVE_POS)
            self.schedule.add(scout)
        self.scout_count = n_scouts
