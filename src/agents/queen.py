from ..config import EGG_INTERVAL, EGGS_PER_INTERVAL_MAX, EGG_INCUBATION_STEPS
from .base import BeeAgent


class QueenAgent(BeeAgent):
    """
    The colony's single queen. Stays in the hive, never moves.

    Every EGG_INTERVAL steps she lays eggs; clutch size scales linearly
    with current nectar stores so egg production slows when food is scarce.
    Eggs are pushed onto Hive.brood as timed entries — no Queen agent on grid.
    """

    def __init__(self, model):
        super().__init__(model)
        self._egg_timer: int = 0

    def step(self) -> None:
        self._egg_timer += 1
        if self._egg_timer >= EGG_INTERVAL:
            self._lay_eggs()
            self._egg_timer = 0

    def _lay_eggs(self) -> None:
        nectar_ratio = self.model.hive.nectar_ratio
        num_eggs = max(1, round(EGGS_PER_INTERVAL_MAX * nectar_ratio))
        hatch_at = self.model.schedule.steps + EGG_INCUBATION_STEPS
        for _ in range(num_eggs):
            self.model.hive.add_egg(hatch_at)
