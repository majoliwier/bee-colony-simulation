from typing import List

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


class BeeAgent(Agent):
    def __init__(self, unique_id, model):
        # Mesa Agent base may call super().__init__ differently across versions;
        # avoid calling super with arguments to remain compatible and set
        # required attributes directly.
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.energy = getattr(model, "initial_energy", 1)

    def step(self):
        # Simple swarm behavior:
        # - prefer to move toward average position of neighbors (cohesion)
        # - with some randomness
        try:
            # get neighboring cell coordinates
            neigh_coords = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            # collect agents in neighborhood
            neighbors = []
            try:
                # MultiGrid: get_cell_list_contents expects iterable of coordinates
                contents = self.model.grid.get_cell_list_contents(neigh_coords)
                # contents may be a flat list of agents
                neighbors = contents
            except Exception:
                # fallback: iterate coordinates
                for c in neigh_coords:
                    for a in self.model.grid.get_cell_list_contents([c]):
                        neighbors.append(a)

            if neighbors:
                # compute average neighbor position
                sx = 0.0
                sy = 0.0
                count = 0
                for a in neighbors:
                    if getattr(a, "pos", None) is None:
                        continue
                    x, y = a.pos
                    sx += x
                    sy += y
                    count += 1
                if count > 0:
                    avgx = sx / count
                    avgy = sy / count
                    # vector toward average
                    cx, cy = self.pos
                    dx = avgx - cx
                    dy = avgy - cy
                    # normalize and apply step size
                    step_size = getattr(self.model, "step_size", 1)
                    # small swarm strength factor
                    strength = getattr(self.model, "swarm_strength", 0.6)
                    # combine cohesion and randomness
                    nx = int(round(cx + strength * dx + (self.random.random() - 0.5)))
                    ny = int(round(cy + strength * dy + (self.random.random() - 0.5)))
                    # clamp to grid
                    nx = max(0, min(self.model.width - 1, nx))
                    ny = max(0, min(self.model.height - 1, ny))
                    new_pos = (nx, ny)
                    self.model.grid.move_agent(self, new_pos)
                    return

            # fallback random move if no neighbors or error
            neigh = neigh_coords
            if neigh:
                new_pos = self.random.choice(neigh)
                self.model.grid.move_agent(self, new_pos)
        except Exception:
            # best-effort: random move
            try:
                neigh = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
                if neigh:
                    new_pos = self.random.choice(neigh)
                    self.model.grid.move_agent(self, new_pos)
            except Exception:
                pass


class BeeModel(Model):
    def __init__(self, width: int = 40, height: int = 40, initial_agents: int = 10, initial_energy: int = 1, swarm_strength: float = 0.6, step_size: int = 1):
        super().__init__()
        self.width = width
        self.height = height
        self.initial_agents = initial_agents
        self.initial_energy = initial_energy
        self.swarm_strength = swarm_strength
        self.step_size = step_size
        # schedule compatible with Mesa DataCollector
        class _Schedule:
            def __init__(self, model):
                self.model = model
                self.agents: List[Agent] = []
                # number of times schedule stepped (compatible with Mesa DataCollector)
                self.steps = 0

            def add(self, agent: Agent):
                self.agents.append(agent)

            def remove(self, agent: Agent):
                try:
                    self.agents.remove(agent)
                except ValueError:
                    pass

            def step(self):
                self.model.random.shuffle(self.agents)
                for a in list(self.agents):
                    a.step()
                # increment steps counter after executing agents
                self.steps += 1

            def get_agent_count(self):
                return len(self.agents)

        self.schedule = _Schedule(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.running = True
        for i in range(self.initial_agents):
            a = BeeAgent(i, self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            self.schedule.add(a)
        # data collector to track agent counts and simple stats
        self.datacollector = DataCollector(
            model_reporters={"AgentCount": lambda m: m.schedule.get_agent_count()},
            agent_reporters={"Energy": lambda a: getattr(a, "energy", 0)},
        )

    def step(self):
        # collect data, then run scheduled agents
        self.datacollector.collect(self)
        self.schedule.step()


def run_mesa_from_cfg(cfg: dict) -> BeeModel:
    grid = cfg.get("grid", {})
    shape = tuple(grid.get("shape", [40, 40]))
    width, height = shape[1], shape[0]
    agents = cfg.get("mesa_agents", {}).get("initial_agents", 10)
    model = BeeModel(width, height, initial_agents=agents)
    steps = int(cfg.get("steps", 10))
    for _ in range(steps):
        model.step()
    return model
