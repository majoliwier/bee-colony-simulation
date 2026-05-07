import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .agents.forager import ForagerAgent, ForagerState
from .config import VIZ_UPDATE_INTERVAL, MAX_NECTAR_STORES

# ── Colours ──────────────────────────────────────────────────────────────────
_HIVE_COLOR  = (1.0, 0.85, 0.0)    # yellow
_PATCH_COLOR = (0.18, 0.65, 0.18)  # green
_BG_COLOR    = (0.93, 0.93, 0.93)  # light grey

_FORAGER_COLORS = {
    ForagerState.RESTING:    "gold",
    ForagerState.SCOUTING:   "darkorange",
    ForagerState.COLLECTING: "limegreen",
    ForagerState.RETURNING:  "crimson",
}


def run_visualization(model, steps: int) -> None:
    """Run the model for `steps` ticks while updating a live matplotlib display."""
    fig, (ax_grid, ax_stats) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.patch.set_facecolor("#1a1a2e")
    plt.ion()

    for step in range(steps):
        model.step()
        if step % VIZ_UPDATE_INTERVAL == 0:
            _draw_grid(ax_grid, model)
            _draw_stats(ax_stats, model)
            fig.suptitle(
                f"Bee Colony Simulation  —  step {model.schedule.steps}",
                color="white", fontsize=13,
            )
            plt.tight_layout(pad=2.0)
            plt.pause(0.01)

    plt.ioff()
    plt.show()


# ── Grid panel ───────────────────────────────────────────────────────────────

def _draw_grid(ax, model) -> None:
    ax.clear()
    ax.set_facecolor(_BG_COLOR)
    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)
    ax.set_aspect("equal")
    ax.set_title("Grid", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    # Hive (3×3 block centred on hive pos)
    hx, hy = model.hive.pos
    ax.add_patch(mpatches.FancyBboxPatch(
        (hx - 1.5, hy - 1.5), 3, 3,
        boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="black", facecolor=_HIVE_COLOR, zorder=2,
    ))

    # Flower patches — radius scales with current nectar level
    for patch in model.flower_patches:
        px, py = patch.pos
        fill = patch.nectar / patch.max_nectar if patch.max_nectar > 0 else 0
        r = 0.15 + 0.45 * fill
        ax.add_patch(mpatches.Circle(
            (px + 0.5, py + 0.5), r,
            color=_PATCH_COLOR, zorder=3,
        ))

    # Foragers — coloured by state
    for agent in model.schedule.agents:
        if isinstance(agent, ForagerAgent) and agent.pos is not None:
            px, py = agent.pos
            color = _FORAGER_COLORS.get(agent.state, "blue")
            ax.plot(px + 0.5, py + 0.5, "o", color=color, markersize=5, zorder=4)

    legend_elements = [
        mpatches.Patch(facecolor=_HIVE_COLOR,  edgecolor="k", label="Hive"),
        mpatches.Patch(facecolor=_PATCH_COLOR,              label="Flower patch"),
        mpatches.Patch(facecolor="gold",                    label="Forager: resting"),
        mpatches.Patch(facecolor="darkorange",              label="Forager: scouting"),
        mpatches.Patch(facecolor="limegreen",               label="Forager: collecting"),
        mpatches.Patch(facecolor="crimson",                 label="Forager: returning"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              facecolor="#222", labelcolor="white", framealpha=0.8)


# ── Stats panel ───────────────────────────────────────────────────────────────

def _draw_stats(ax, model) -> None:
    ax.clear()
    ax.set_facecolor("#1a1a2e")
    ax.set_axis_off()

    nurses   = sum(1 for a in model.schedule.agents if a.__class__.__name__ == "NurseAgent")
    foragers = sum(1 for a in model.schedule.agents if a.__class__.__name__ == "ForagerAgent")

    text = (
        f"Step      {model.schedule.steps:>6}\n\n"
        f"Nectar    {model.hive.nectar:>6.0f} / {MAX_NECTAR_STORES:.0f}\n"
        f"Brood     {model.hive.brood_count:>6}\n\n"
        f"Nurses    {nurses:>6}\n"
        f"Foragers  {foragers:>6}\n"
    )
    ax.text(
        0.08, 0.90, text,
        transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        fontfamily="monospace", color="white",
        bbox=dict(boxstyle="round", facecolor="#2a2a4a", alpha=0.9),
    )
    ax.set_title("Colony stats", color="white", fontsize=10)

    # Nectar history chart
    data = model.datacollector.get_model_vars_dataframe()
    if len(data) > 1:
        ax_sub = ax.inset_axes([0.05, 0.05, 0.9, 0.38])
        ax_sub.set_facecolor("#0d0d1a")
        ax_sub.plot(data["Nectar"].values,   color="gold",       lw=1.5, label="Nectar")
        ax_sub.plot(data["Nurses"].values,   color="dodgerblue", lw=1.2, label="Nurses")
        ax_sub.plot(data["Foragers"].values, color="darkorange",  lw=1.2, label="Foragers")
        ax_sub.set_title("Trends", color="white", fontsize=8)
        ax_sub.tick_params(colors="white", labelsize=6)
        for spine in ax_sub.spines.values():
            spine.set_edgecolor("#555")
        ax_sub.legend(fontsize=6, facecolor="#222", labelcolor="white", framealpha=0.8)
