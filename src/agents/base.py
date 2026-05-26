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
