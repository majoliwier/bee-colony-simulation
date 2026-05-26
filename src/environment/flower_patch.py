from ..config import PATCH_MAX_NECTAR, PATCH_REGEN_RATE


class FlowerPatch:
    """
    Passive element representing a flower patch on the grid.
    Nectar regenerates each step; quality multiplies the effective yield.

    Not a Mesa agent — stored in BeeColonyModel.flower_patches
    and stepped by the model each tick.
    """

    def __init__(
        self,
        pos: tuple,
        max_nectar: float = PATCH_MAX_NECTAR,
        quality: float = 1.0,
        regen_rate: float = PATCH_REGEN_RATE,
    ):
        self.pos = pos
        self.max_nectar = max_nectar
        self.quality = quality      # multiplier applied to raw collected nectar
        self.regen_rate = regen_rate
        self.nectar = max_nectar * 0.8  # start mostly full

    def collect(self, requested: float) -> float:
        """
        Extract up to `requested` units from this patch.
        Returns actual amount * quality (richer patches yield more effective nectar).
        """
        raw = min(self.nectar, requested)
        self.nectar -= raw
        return raw * self.quality

    def step(self) -> None:
        self.nectar = min(self.max_nectar, self.nectar + self.regen_rate)

    @property
    def is_depleted(self) -> bool:
        return self.nectar < 0.1
