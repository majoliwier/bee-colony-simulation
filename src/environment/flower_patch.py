from ..config import PATCH_MAX_NECTAR, PATCH_REGEN_RATE, PATCH_LIFETIME_NECTAR


class FlowerPatch:
    """
    Passive element representing a flower patch on the grid.
    Nectar regenerates each step up to max_nectar, but the patch has a finite
    lifetime budget (PATCH_LIFETIME_NECTAR total nectar yielded).  Once the
    budget is spent the patch is permanently exhausted and stops regenerating.

    Not a Mesa agent — stored in BeeColonyModel.flower_patches
    and stepped by the model each tick.
    """

    def __init__(
        self,
        pos: tuple,
        max_nectar: float = PATCH_MAX_NECTAR,
        quality: float = 1.0,
        regen_rate: float = PATCH_REGEN_RATE,
        lifetime_nectar: float = PATCH_LIFETIME_NECTAR,
    ):
        self.pos = pos
        self.max_nectar = max_nectar
        self.quality = quality
        self.regen_rate = regen_rate
        self._lifetime_budget = lifetime_nectar   # total raw nectar this patch can ever give
        self._total_collected = 0.0
        self.exhausted = False                    # True once lifetime_budget is spent
        self.nectar = max_nectar * 0.8            # start mostly full

    def collect(self, requested: float) -> float:
        """
        Extract up to `requested` units.  Returns actual amount × quality.
        Marks the patch exhausted if the lifetime budget is spent.
        """
        raw = min(self.nectar, requested)
        self.nectar -= raw
        self._total_collected += raw
        if self._total_collected >= self._lifetime_budget:
            self.nectar = 0.0
            self.exhausted = True
        return raw * self.quality

    def step(self) -> None:
        if not self.exhausted:
            self.nectar = min(self.max_nectar, self.nectar + self.regen_rate)

    @property
    def is_depleted(self) -> bool:
        return self.nectar < 0.1
