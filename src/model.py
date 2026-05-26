from __future__ import annotations

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from .config import (
    GRID_WIDTH, GRID_HEIGHT,
    HIVE_POS, INITIAL_NECTAR,
    INITIAL_NURSES, INITIAL_FORAGERS,
    NUM_FLOWER_PATCHES,
    PATCH_MAX_NECTAR, PATCH_REGEN_RATE, PATCH_QUALITY_RANGE, MIN_PATCH_DISTANCE,
)
from .agents.queen import QueenAgent
from .agents.nurse import NurseAgent
from .agents.forager import ForagerAgent
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
        # O(1) patch lookup keyed by grid position — kept in sync with flower_patches list.
        self._patch_index: dict[tuple, FlowerPatch] = {}
        # Agent-type counters — kept in sync manually so callers avoid O(n) scans.
        self.nurse_count: int = 0
        self.forager_count: int = 0

        self._place_flower_patches(num_patches)
        self._spawn_initial_agents(initial_nurses, initial_foragers)

        self.datacollector = DataCollector(
            model_reporters={
                "Nectar":   lambda m: round(m.hive.nectar, 1),
                "Nurses":   lambda m: m.nurse_count,
                "Foragers": lambda m: m.forager_count,
                "Brood":    lambda m: m.hive.brood_count,
            }
        )

    # ── Per-step logic ───────────────────────────────────────────────────────

    def step(self) -> None:
        # Hatch eggs whose incubation period ended — each produces one nurse.
        new_nurses = self.hive.hatch(self.schedule.steps)
        for _ in range(new_nurses):
            nurse = NurseAgent(self)
            self.schedule.add(nurse)
            self.nurse_count += 1

        for patch in self.flower_patches:
            patch.step()

        self.datacollector.collect(self)
        self.schedule.step()

    # ── Query helpers ────────────────────────────────────────────────────────

    def get_patch_at(self, pos: tuple) -> FlowerPatch | None:
        """Return the FlowerPatch at `pos`, or None. O(1) via position index."""
        return self._patch_index.get(pos)

    # ── Initialisation ───────────────────────────────────────────────────────

    def _place_flower_patches(self, n: int) -> None:
        hx, hy = HIVE_POS
        # Clamp sampling range to valid interior cells (always at least one cell wide).
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

    def _spawn_initial_agents(self, n_nurses: int, n_foragers: int) -> None:
        # Queen — lives inside hive, not placed on grid
        self.schedule.add(QueenAgent(self))

        for _ in range(n_nurses):
            self.schedule.add(NurseAgent(self))
        self.nurse_count = n_nurses

        for _ in range(n_foragers):
            forager = ForagerAgent(self)
            self.grid.place_agent(forager, HIVE_POS)
            self.schedule.add(forager)
        self.forager_count = n_foragers
