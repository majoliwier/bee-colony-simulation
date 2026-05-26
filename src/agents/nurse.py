from ..config import (
    NURSE_FEEDING_RATE,
    NURSE_TO_FORAGER_AGE,
    NURSE_FORAGER_THRESHOLD_RANGE,
    MAX_NURSE_AGE,
)
from .base import BeeAgent


class NurseAgent(BeeAgent):
    """
    Young worker bee caring for brood inside the hive.
    Consumes a small amount of nectar each step to feed larvae.

    Role switching (HoPoMo — Schmickl & Crailsheim 2007):
    - Natural transition: age reaches NURSE_TO_FORAGER_AGE.
    - Stress transition: colony nectar deficit exceeds this nurse's personal
      threshold (drawn randomly at birth to model population heterogeneity).
      Hungry colonies push nurses into foraging earlier, automatically
      balancing the workforce without any central control.

    Nurses are NOT placed on the grid — they live logically inside the hive.
    """

    def __init__(self, model):
        super().__init__(model)
        self.age: int = 0
        lo, hi = NURSE_FORAGER_THRESHOLD_RANGE
        self.forager_threshold: float = model.random.uniform(lo, hi)

    def step(self) -> None:
        self.age += 1
        self.model.hive.consume(NURSE_FEEDING_RATE)

        if self.age >= MAX_NURSE_AGE:
            self._die()
            return

        if self._should_become_forager():
            self._become_forager()

    def _should_become_forager(self) -> bool:
        return (
            self.age >= NURSE_TO_FORAGER_AGE
            or self.model.hive.nectar_deficit > self.forager_threshold
        )

    def _die(self) -> None:
        self.model.schedule.remove(self)
        self.model.nurse_count -= 1

    def _become_forager(self) -> None:
        from .forager import ForagerAgent  # local import avoids circular dependency

        forager = ForagerAgent(self.model)
        self.model.grid.place_agent(forager, self.model.hive.pos)
        self.model.schedule.add(forager)
        self.model.schedule.remove(self)
        self.model.nurse_count -= 1
        self.model.forager_count += 1
        # Note: nurse was never placed on the grid, so no grid.remove_agent needed.
