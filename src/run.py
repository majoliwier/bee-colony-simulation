try:
    from mesa.visualization.ModularVisualization import ModularServer
    from mesa.visualization.modules import CanvasGrid, ChartModule
    from mesa.visualization.UserParam import UserSettableParameter
except Exception as e:
    raise ImportError("ModularServer UI requires an older Mesa version (ModularVisualization). "
                      "Install mesa==0.8.9 and retry. Original error: " + str(e))

from .mesa_model import BeeModel


def agent_portrayal(agent):
    if agent is None:
        return
    portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 1, "Color": "orange"}
    return portrayal


def make_server():
    model_params = {
        "width": UserSettableParameter("number", "Width", 40, 10, 200, 1),
        "height": UserSettableParameter("number", "Height", 40, 10, 200, 1),
        "initial_agents": UserSettableParameter("slider", "Initial agents", 80, 1, 200, 1),
        "swarm_strength": UserSettableParameter("slider", "Swarm strength", 0.6, 0.0, 1.0, 0.05),
    }

    grid = CanvasGrid(agent_portrayal, 40, 40, 400, 400)
    chart = ChartModule([{"Label": "AgentCount", "Color": "#AA0000"}], data_collector_name="datacollector")

    server = ModularServer(BeeModel, [grid, chart], "Bee Swarm", model_params)
    server.port = 8521
    return server


if __name__ == "__main__":
    make_server().launch()
