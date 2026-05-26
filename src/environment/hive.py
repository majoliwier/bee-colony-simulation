from __future__ import annotations

from ..config import MAX_NECTAR_STORES


class Hive:
    """
    Passive data structure representing the single beehive.
    Tracks nectar stores and the brood (egg) incubation queue.

    Not a Mesa agent — owned and stepped directly by BeeColonyModel.
    """

    def __init__(self, pos: tuple, initial_nectar: float):
        self.pos = pos
        self.nectar = initial_nectar
        # Each entry is the simulation step at which that egg hatches into a nurse.
        self._brood: list[int] = []

    # ── Nectar ───────────────────────────────────────────────────────────────

    def deposit(self, amount: float) -> None:
        self.nectar = min(MAX_NECTAR_STORES, self.nectar + amount)

    def consume(self, amount: float) -> float:
        """Consume up to `amount` nectar. Returns how much was actually consumed."""
        consumed = min(self.nectar, amount)
        self.nectar -= consumed
        return consumed

    @property
    def nectar_ratio(self) -> float:
        """0.0 = empty, 1.0 = full."""
        return self.nectar / MAX_NECTAR_STORES

    @property
    def nectar_deficit(self) -> float:
        """Inverse of nectar_ratio. Used by HoPoMo nurses to decide role switch."""
        return 1.0 - self.nectar_ratio

    # ── Brood ────────────────────────────────────────────────────────────────

    def add_egg(self, hatch_step: int) -> None:
        self._brood.append(hatch_step)

    def hatch(self, current_step: int) -> int:
        """Remove and return the number of eggs ready to hatch at `current_step`."""
        remaining, ready = [], 0
        for s in self._brood:
            if s <= current_step:
                ready += 1
            else:
                remaining.append(s)
        self._brood = remaining
        return ready

    @property
    def brood_count(self) -> int:
        return len(self._brood)
