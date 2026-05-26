from ..config import SCOUT_MAX_ENERGY, FORAGER_ENERGY_COST_PER_STEP
from .base import BeeAgent


class ScoutState:
    SCOUTING  = "SCOUTING"
    RETURNING = "RETURNING"


class ScoutAgent(BeeAgent):
    """
    Dedicated explorer bee. Never collects nectar.

    FSM:
        SCOUTING (random walk) → (patch found) → RETURNING → (at hive) → SCOUTING

    On hive arrival after finding a patch, triggers a waggle dance to recruit
    resting foragers. Scouts die when energy reaches zero.
    """

    def __init__(self, model):
        super().__init__(model)
        self.state: str = ScoutState.SCOUTING
        self.target_patch = None
        self.energy: float = float(SCOUT_MAX_ENERGY)

    def step(self) -> None:
        self.energy -= FORAGER_ENERGY_COST_PER_STEP
        if self.energy <= 0:
            self._die()
            return

        if self.state == ScoutState.SCOUTING:
            self._step_scouting()
        elif self.state == ScoutState.RETURNING:
            self._step_returning()

    def _step_scouting(self) -> None:
        self._random_move()
        patch = self.model.get_patch_at(self.pos)
        if patch and not patch.is_depleted:
            self.target_patch = patch
            self.state = ScoutState.RETURNING

    def _step_returning(self) -> None:
        if self.target_patch is None or self.target_patch.is_depleted:
            self.target_patch = None
            self.state = ScoutState.SCOUTING
            return
        if self.pos == self.model.hive.pos:
            self.model.perform_waggle_dance(self.target_patch)
            self.target_patch = None
            self.state = ScoutState.SCOUTING
        else:
            self._move_toward(self.model.hive.pos)

    def _die(self) -> None:
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)
        self.model.scout_count -= 1

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
