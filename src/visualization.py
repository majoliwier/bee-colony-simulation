import itertools
import math
import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .agents.forager import ForagerAgent, ForagerState
from .config import VIZ_UPDATE_INTERVAL, MAX_NECTAR_STORES, DEFAULT_STEPS

# ── Colour palette ────────────────────────────────────────────────────────────
_HIVE_COLOR  = (1.0, 0.85, 0.0)
_PATCH_COLOR = (0.18, 0.65, 0.18)
_BG_COLOR    = (0.93, 0.93, 0.93)

_NURSE_COLOR = "dodgerblue"
_QUEEN_COLOR = "gold"

_FORAGER_COLORS = {
    ForagerState.RESTING:    "#b8860b",   # dark gold  (at hive, idle)
    ForagerState.SCOUTING:   "darkorange",
    ForagerState.COLLECTING: "limegreen",
    ForagerState.RETURNING:  "crimson",
}


def run_visualization(model, steps: int) -> None:
    """
    Run the model for `steps` ticks with a live matplotlib display.
    Press Space to pause / resume.
    """
    fig, (ax_grid, ax_stats) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [2, 1]},
        constrained_layout=True,   # computed once; avoids tight_layout() in the loop
    )
    fig.patch.set_facecolor("#1a1a2e")

    paused = [False]

    def _on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]
            label = "PAUSED — Space to resume" if paused[0] else _title(model)
            fig.suptitle(label, color="white", fontsize=13)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.ion()

    tick = range(steps) if steps is not None else itertools.count()
    for step in tick:
        while paused[0]:
            plt.pause(0.1)   # keep event loop alive while paused

        model.step()

        if step % VIZ_UPDATE_INTERVAL == 0:
            _draw_grid(ax_grid, model)
            _draw_stats(ax_stats, model)
            fig.suptitle(_title(model), color="white", fontsize=13)
            plt.pause(0.01)

    plt.ioff()
    plt.show()


def record_visualization(model, steps: int, output_path: str, fps: int = 30) -> None:
    """
    Run the model headlessly and write every rendered frame to an MP4.
    Requires: pip install imageio imageio-ffmpeg
    """
    try:
        import imageio
    except ImportError:
        raise SystemExit("imageio not found — run: pip install imageio imageio-ffmpeg")

    import numpy as np

    steps = steps if steps is not None else DEFAULT_STEPS

    fig, (ax_grid, ax_stats) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [2, 1]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#1a1a2e")
    fig.canvas.draw()  # initialise renderer before grabbing pixels

    print(f"Recording {steps} steps → {output_path}  (fps={fps})")

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    try:
        for step in range(steps):
            model.step()
            _draw_grid(ax_grid, model)
            _draw_stats(ax_stats, model)
            fig.suptitle(
                f"Bee Colony Simulation  —  step {model.schedule.steps}",
                color="white", fontsize=13,
            )
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
            if step % 50 == 0:
                print(f"  step {step}/{steps}", end="\r", flush=True)
    finally:
        writer.close()
        plt.close(fig)

    print(f"\nSaved: {output_path}")


def _title(model) -> str:
    return f"Bee Colony Simulation  —  step {model.schedule.steps}  [Space = pause]"


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

    hx, hy = model.hive.pos

    # Hive block
    ax.add_patch(mpatches.FancyBboxPatch(
        (hx - 1.5, hy - 1.5), 3, 3,
        boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="black", facecolor=_HIVE_COLOR, zorder=2,
    ))

    # Queen — star at hive centre
    ax.plot(hx + 0.5, hy + 0.9, "*", color=_QUEEN_COLOR,
            markersize=11, zorder=6, markeredgecolor="black", markeredgewidth=0.4)

    # Nurses — blue dots arranged in rows inside hive block
    _draw_nurses(ax, model, hx, hy)

    # Flower patches — radius scales with remaining nectar
    for patch in model.flower_patches:
        px, py = patch.pos
        fill = patch.nectar / patch.max_nectar if patch.max_nectar > 0 else 0
        r = 0.15 + 0.45 * fill
        ax.add_patch(mpatches.Circle(
            (px + 0.5, py + 0.5), r, color=_PATCH_COLOR, zorder=3,
        ))

    # Foragers — coloured by FSM state
    for agent in model.schedule.agents:
        if isinstance(agent, ForagerAgent) and agent.pos is not None:
            px, py = agent.pos
            color = _FORAGER_COLORS.get(agent.state, "blue")
            ax.plot(px + 0.5, py + 0.5, "o", color=color, markersize=5, zorder=4)

    ax.legend(handles=_grid_legend(), loc="upper left", fontsize=7,
              facecolor="#222", labelcolor="white", framealpha=0.8)


def _draw_nurses(ax, model, hx: int, hy: int) -> None:
    """Draw nurse dots inside the hive block (max 12 shown; count label if more)."""
    nurses = model.nurse_count
    visible = min(nurses, 12)
    cols = 4
    for i in range(visible):
        row, col = divmod(i, cols)
        x = hx - 1.1 + col * 0.75 + 0.5
        y = hy - 0.6 + row * 0.65
        ax.plot(x, y, "o", color=_NURSE_COLOR, markersize=3.5, zorder=5)
    if nurses > visible:
        ax.text(hx + 0.5, hy - 1.3, f"+{nurses - visible}",
                color=_NURSE_COLOR, fontsize=6, ha="center", zorder=6)


def _grid_legend() -> list:
    return [
        mpatches.Patch(facecolor=_HIVE_COLOR,   edgecolor="k", label="Hive"),
        mpatches.Patch(facecolor=_PATCH_COLOR,               label="Flower patch"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=_QUEEN_COLOR,
                   markersize=10, label="Queen"),
        mpatches.Patch(facecolor=_NURSE_COLOR,               label="Nurse"),
        mpatches.Patch(facecolor="darkorange",               label="Forager: scouting"),
        mpatches.Patch(facecolor="limegreen",                label="Forager: collecting"),
        mpatches.Patch(facecolor="crimson",                  label="Forager: returning"),
        mpatches.Patch(facecolor="#b8860b",                  label="Forager: resting"),
    ]


# ── Stats panel ───────────────────────────────────────────────────────────────

def _draw_stats(ax, model) -> None:
    ax.clear()
    ax.set_facecolor("#1a1a2e")
    ax.set_axis_off()
    ax.set_title("Colony stats", color="white", fontsize=10)

    nurses   = model.nurse_count
    foragers = model.forager_count

    text = (
        f"Step      {model.schedule.steps:>6}\n\n"
        f"Nectar    {model.hive.nectar:>6.0f} / {MAX_NECTAR_STORES:.0f}\n"
        f"Brood     {model.hive.brood_count:>6}\n\n"
        f"Nurses    {nurses:>6}\n"
        f"Foragers  {foragers:>6}\n"
    )
    ax.text(
        0.08, 0.92, text,
        transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        fontfamily="monospace", color="white",
        bbox=dict(boxstyle="round", facecolor="#2a2a4a", alpha=0.9),
    )

    data = model.datacollector.get_model_vars_dataframe()
    if len(data) > 1:
        ax_sub = ax.inset_axes([0.05, 0.05, 0.9, 0.38])
        ax_sub.set_facecolor("#0d0d1a")
        ax_sub.plot(data["Nectar"].values,   color="gold",        lw=1.5, label="Nectar")
        ax_sub.plot(data["Nurses"].values,   color=_NURSE_COLOR,  lw=1.2, label="Nurses")
        ax_sub.plot(data["Foragers"].values, color="darkorange",  lw=1.2, label="Foragers")
        ax_sub.set_title("Trends", color="white", fontsize=8)
        ax_sub.tick_params(colors="white", labelsize=6)
        for spine in ax_sub.spines.values():
            spine.set_edgecolor("#555")
        ax_sub.legend(fontsize=6, facecolor="#222", labelcolor="white", framealpha=0.8)
