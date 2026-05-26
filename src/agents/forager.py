from enum import Enum, auto

from ..config import (
    FORAGER_REST_DURATION, FORAGER_LOAD_CAPACITY, COLLECTING_STEPS,
    FORAGER_MAX_ENERGY, FORAGER_ENERGY_COST_PER_STEP,
)

_NECTAR_PER_COLLECT_STEP = FORAGER_LOAD_CAPACITY / COLLECTING_STEPS
from .base import BeeAgent


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
        self._random_move()
        patch = self.model.get_patch_at(self.pos)
        if patch and not patch.is_depleted:
            self.target_patch = patch
            self._collect_timer = 0
            self.state = ForagerState.COLLECTING

    def _step_flying_to_patch(self) -> None:
        if self.target_patch is None or self.target_patch.is_depleted:
            # Patch gone — fall back to scouting
            self.target_patch = None
            self.state = ForagerState.SCOUTING
            return
        if self.pos == self.target_patch.pos:
            self._collect_timer = 0
            self.state = ForagerState.COLLECTING
        else:
            self._move_toward(self.target_patch.pos)

    def _step_collecting(self) -> None:
        self._collect_timer += 1
        if self.target_patch and not self.target_patch.is_depleted:
            collected = self.target_patch.collect(_NECTAR_PER_COLLECT_STEP)
            self.nectar_load = min(FORAGER_LOAD_CAPACITY, self.nectar_load + collected)

        patch_done = self.target_patch is None or self.target_patch.is_depleted
        if self._collect_timer >= COLLECTING_STEPS or patch_done:
            self.state = ForagerState.RETURNING

    def _step_returning(self) -> None:
        if self.pos == self.model.hive.pos:
            self.model.hive.deposit(self.nectar_load)
            self.nectar_load = 0.0
            if self.target_patch is not None:
                self.model.perform_waggle_dance(self.target_patch)
            self.target_patch = None
            self.state = ForagerState.RESTING
        else:
            self._move_toward(self.model.hive.pos)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _die(self) -> None:
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)
        self.model.forager_count -= 1

    # ── Movement ─────────────────────────────────────────────────────────────

    def _random_move(self) -> None:
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        self.model.grid.move_agent(self, self.random.choice(neighbors))

    def _move_toward(self, target: tuple) -> None:
        cx, cy = self.pos
        tx, ty = target
        dx = (1 if tx > cx else -1) if tx != cx else 0
        dy = (1 if ty > cy else -1) if ty != cy else 0
        new_pos = (
            max(0, min(self.model.width - 1, cx + dx)),
            max(0, min(self.model.height - 1, cy + dy)),
        )
        self.model.grid.move_agent(self, new_pos)
