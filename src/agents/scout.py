from enum import Enum, auto

from ..config import SCOUT_MAX_ENERGY, FORAGER_ENERGY_COST_PER_STEP, SMELL_RADIUS
from .base import BeeAgent


class ScoutState(Enum):
    SCOUTING  = auto()
    RETURNING = auto()


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
        self.state: ScoutState = ScoutState.SCOUTING
        self.target_patch = None
        self.energy: float = float(SCOUT_MAX_ENERGY)
        self._found_at_step: int = 0

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
        self._biased_move()
        patch = self.model.get_patch_near(self.pos, SMELL_RADIUS)
        if patch and not patch.is_depleted:
            self.model.record_patch_discovery(patch, "scout")
            self.target_patch = patch
            self._found_at_step = self.model.schedule.steps
            self.state = ScoutState.RETURNING

    def _step_returning(self) -> None:
        if self.target_patch is None or self.target_patch.is_depleted:
            self.target_patch = None
            self.state = ScoutState.SCOUTING
            return
        if self.pos == self.model.hive.pos:
            self.model.perform_waggle_dance(self.target_patch, caller="scout", found_step=self._found_at_step)
            self.target_patch = None
            self.state = ScoutState.SCOUTING
        else:
            self._move_toward(self.model.hive.pos)

    def _die(self) -> None:
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)
        self.model.scout_count -= 1
        from ..config import INITIAL_SCOUTS, HIVE_POS
        if self.model.scout_count < INITIAL_SCOUTS:
            replacement = ScoutAgent(self.model)
            self.model.grid.place_agent(replacement, HIVE_POS)
            self.model.schedule.add(replacement)
            self.model.scout_count += 1
