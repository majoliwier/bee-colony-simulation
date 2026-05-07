from enum import Enum, auto

from ..config import FORAGER_REST_DURATION, FORAGER_LOAD_CAPACITY, COLLECTING_STEPS
from .base import BeeAgent


class ForagerState(Enum):
    RESTING = auto()
    SCOUTING = auto()
    COLLECTING = auto()
    RETURNING = auto()


class ForagerAgent(BeeAgent):
    """
    Worker bee that collects nectar from flower patches.

    FSM:
        RESTING ──► SCOUTING ──► COLLECTING ──► RETURNING ──► RESTING

    RESTING:    Waits in hive for rest_timer to expire.
    SCOUTING:   Random walk until it lands on a non-depleted flower patch.
    COLLECTING: Stays on patch for up to COLLECTING_STEPS, filling nectar_load.
    RETURNING:  Moves directly toward hive; deposits load on arrival.

    Foragers are placed on the grid and move around it.
    Future extension points: waggle dance recruitment, pheromone deposition.
    """

    def __init__(self, model):
        super().__init__(model)
        self.state: ForagerState = ForagerState.RESTING
        self.nectar_load: float = 0.0
        self.target_patch = None       # FlowerPatch reference while COLLECTING/RETURNING
        self._rest_timer: int = 0
        self._collect_timer: int = 0

    # ── Main dispatch ────────────────────────────────────────────────────────

    def step(self) -> None:
        {
            ForagerState.RESTING:    self._step_resting,
            ForagerState.SCOUTING:   self._step_scouting,
            ForagerState.COLLECTING: self._step_collecting,
            ForagerState.RETURNING:  self._step_returning,
        }[self.state]()

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

    def _step_collecting(self) -> None:
        self._collect_timer += 1
        if self.target_patch and not self.target_patch.is_depleted:
            per_step = FORAGER_LOAD_CAPACITY / COLLECTING_STEPS
            collected = self.target_patch.collect(per_step)
            self.nectar_load = min(FORAGER_LOAD_CAPACITY, self.nectar_load + collected)

        patch_done = self.target_patch is None or self.target_patch.is_depleted
        if self._collect_timer >= COLLECTING_STEPS or patch_done:
            self.state = ForagerState.RETURNING

    def _step_returning(self) -> None:
        if self.pos == self.model.hive.pos:
            self.model.hive.deposit(self.nectar_load)
            self.nectar_load = 0.0
            self.target_patch = None
            self.state = ForagerState.RESTING
        else:
            self._move_toward(self.model.hive.pos)

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
