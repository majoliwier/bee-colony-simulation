from enum import Enum, auto

from ..config import (
    FORAGER_REST_DURATION, FORAGER_LOAD_CAPACITY, COLLECTING_STEPS,
    FORAGER_MAX_ENERGY, FORAGER_ENERGY_COST_PER_STEP,
    TRAIL_DEPOSIT_STRENGTH, LOCAL_SEARCH_STEPS, SMELL_RADIUS,
)
from .base import BeeAgent

_NECTAR_PER_COLLECT_STEP = FORAGER_LOAD_CAPACITY / COLLECTING_STEPS


class ForagerState(Enum):
    RESTING         = auto()
    SCOUTING        = auto()
    FLYING_TO_PATCH = auto()  # recruited via waggle dance; knows target_patch
    COLLECTING      = auto()
    RETURNING       = auto()


class ForagerAgent(BeeAgent):
    """
    Worker bee that collects nectar from flower patches.

    FSM:
        RESTING ──► SCOUTING ──► COLLECTING ──► RETURNING ──► RESTING
        RESTING (recruited) ──► FLYING_TO_PATCH ──► COLLECTING ──► RETURNING

    Energy depletes every step outside of RESTING; forager dies at zero.
    On arrival at hive after RETURNING, performs waggle dance to recruit others.
    """

    def __init__(self, model):
        super().__init__(model)
        self.state: ForagerState = ForagerState.RESTING
        self.nectar_load: float = 0.0
        self.target_patch = None
        self._rest_timer: int = 0
        self._collect_timer: int = 0
        self.energy: float = float(FORAGER_MAX_ENERGY)
        self._scout_log_entry: dict | None = None  # set when recruited by a waggle dance
        self._found_at_step: int = 0
        self.dance_target_pos: tuple | None = None    # noisy landing zone from waggle dance
        self._local_search_remaining: int = 0        # steps left in post-miss local search

    # ── Main dispatch ────────────────────────────────────────────────────────

    def step(self) -> None:
        if self.state != ForagerState.RESTING:
            self.energy -= FORAGER_ENERGY_COST_PER_STEP
            if self.energy <= 0:
                self._die()
                return

        if self.state == ForagerState.RESTING:
            self._step_resting()
        elif self.state == ForagerState.SCOUTING:
            self._step_scouting()
        elif self.state == ForagerState.FLYING_TO_PATCH:
            self._step_flying_to_patch()
        elif self.state == ForagerState.COLLECTING:
            self._step_collecting()
        elif self.state == ForagerState.RETURNING:
            self._step_returning()

    # ── State handlers ───────────────────────────────────────────────────────

    def _step_resting(self) -> None:
        self._rest_timer += 1
        if self._rest_timer >= FORAGER_REST_DURATION:
            self._rest_timer = 0
            self.state = ForagerState.SCOUTING

    def _step_scouting(self) -> None:
        self._biased_move()
        patch = self.model.get_patch_near(self.pos, SMELL_RADIUS)
        if patch and not patch.is_depleted:
            self.model.record_patch_discovery(patch, "forager")
            self.target_patch = patch
            self._found_at_step = self.model.schedule.steps
            if self.pos == patch.pos:
                self._collect_timer = 0
                self.state = ForagerState.COLLECTING
            else:
                self.state = ForagerState.FLYING_TO_PATCH

    def _step_flying_to_patch(self) -> None:
        if self.target_patch is None or self.target_patch.is_depleted:
            self._scout_log_entry = None
            self.target_patch = None
            self.dance_target_pos = None
            self._local_search_remaining = 0
            self.state = ForagerState.SCOUTING
            return

        # Pheromone-guided local search after missing the dance landing zone.
        if self._local_search_remaining > 0:
            self._local_search_remaining -= 1
            self._biased_move()
            patch = self.model.get_patch_near(self.pos, SMELL_RADIUS)
            if patch and not patch.is_depleted:
                self.model.record_patch_discovery(patch, "forager")
                self._scout_log_entry = None
                self.target_patch = patch
                self._local_search_remaining = 0
                if self.pos == patch.pos:
                    self._collect_timer = 0
                    self.state = ForagerState.COLLECTING
            elif self._local_search_remaining == 0:
                self.target_patch = None
                self.state = ForagerState.RETURNING
            return

        # Fly to the dance-encoded approximate landing zone first.
        if self.dance_target_pos is not None:
            if self.pos == self.dance_target_pos:
                self.dance_target_pos = None
                if self.pos == self.target_patch.pos:
                    if self._scout_log_entry is not None:
                        self._scout_log_entry["arrived_step"] = self.model.schedule.steps
                        self._scout_log_entry = None
                    self._collect_timer = 0
                    self.state = ForagerState.COLLECTING
                else:
                    # Dance noise caused overshoot; begin local search.
                    self._scout_log_entry = None
                    self._local_search_remaining = LOCAL_SEARCH_STEPS
            else:
                self._move_toward(self.dance_target_pos)
                self._deposit_trail()
            return

        # Direct flight to known patch position.
        if self.pos == self.target_patch.pos:
            if self._scout_log_entry is not None:
                self._scout_log_entry["arrived_step"] = self.model.schedule.steps
                self._scout_log_entry = None
            self._collect_timer = 0
            self.state = ForagerState.COLLECTING
        else:
            self._move_toward(self.target_patch.pos)
            self._deposit_trail()

    def _step_collecting(self) -> None:
        self._collect_timer += 1
        if self.target_patch and not self.target_patch.is_depleted:
            collected = self.target_patch.collect(_NECTAR_PER_COLLECT_STEP)
            self.nectar_load = min(FORAGER_LOAD_CAPACITY, self.nectar_load + collected)
            self._deposit_trail()
            self._broadcast_patch_marker()

        patch_done = self.target_patch is None or self.target_patch.is_depleted
        if self._collect_timer >= COLLECTING_STEPS or patch_done:
            self.state = ForagerState.RETURNING

    def _step_returning(self) -> None:
        if self.pos == self.model.hive.pos:
            self.model.hive.deposit(self.nectar_load)
            self.nectar_load = 0.0
            self.energy = float(FORAGER_MAX_ENERGY)
            if self.target_patch is not None and not self.target_patch.exhausted:
                self.model.perform_waggle_dance(self.target_patch, found_step=self._found_at_step)
            self.target_patch = None
            self.state = ForagerState.RESTING
        else:
            self._move_toward(self.model.hive.pos)

    def _deposit_trail(self) -> None:
        if self.target_patch is None or not self.model.use_pheromones:
            return
        deposit = self.model._profitability_ratio(self.target_patch) * self.model.trail_deposit_strength
        x, y = self.pos
        self.model.pheromones[x, y] = min(1.0, self.model.pheromones[x, y] + deposit)

    def _broadcast_patch_marker(self) -> None:
        # Diffusion alone is too slow to spread the signal to a dance landing zone
        # 2-4 cells away, so deposit a weaker halo around the patch directly.

        if self.target_patch is None or not self.model.use_pheromones:
            return
        halo = self.model._profitability_ratio(self.target_patch) * self.model.trail_deposit_strength * 0.35
        if halo <= 0:
            return
        for nx, ny in self.model.grid.get_neighborhood(
            self.target_patch.pos, moore=True, include_center=False
        ):
            self.model.pheromones[nx, ny] = min(1.0, self.model.pheromones[nx, ny] + halo)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _die(self) -> None:
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)
        self.model.forager_count -= 1

