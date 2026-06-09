import inspect
import mesa

# Check the actual Agent signature rather than guessing from the version number.
# Mesa < 2.3 requires Agent.__init__(unique_id, model).
# Mesa >= 2.3 removed unique_id (auto-assigned internally).
_AGENT_NEEDS_UNIQUE_ID = "unique_id" in inspect.signature(mesa.Agent.__init__).parameters


class BeeAgent(mesa.Agent):
    """
    Thin base class that handles the Mesa < 2.3 / >= 2.3 Agent init API difference.
    All project agents inherit from this instead of mesa.Agent directly.
    """

    def __init__(self, model):
        if _AGENT_NEEDS_UNIQUE_ID:
            super().__init__(model.next_id(), model)
        else:
            super().__init__(model)

    # ── Shared movement helpers (used by ForagerAgent and ScoutAgent) ─────────

    def _random_move(self) -> None:
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        self.model.grid.move_agent(self, self.random.choice(neighbors))

    def _biased_move(self) -> None:
        if not self.model.use_pheromones:
            self._random_move()
            return
        from ..config import PHEROMONE_BIAS
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        ph = self.model.pheromones
        weights = [PHEROMONE_BIAS * float(ph[nx, ny]) + (1.0 - PHEROMONE_BIAS)
                   for nx, ny in neighbors]
        self.model.grid.move_agent(self, self.random.choices(neighbors, weights=weights)[0])

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
