import itertools
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.widgets as mwidgets

from .agents.forager import ForagerAgent, ForagerState
from .agents.scout import ScoutAgent, ScoutState
from .config import VIZ_UPDATE_INTERVAL, MAX_NECTAR_STORES, DEFAULT_STEPS

# ── Colour palette ────────────────────────────────────────────────────────────
_HIVE_COLOR  = (1.0, 0.85, 0.0)
_PATCH_COLOR = (0.18, 0.65, 0.18)
_BG_COLOR    = (0.93, 0.93, 0.93)

_NURSE_COLOR  = "dodgerblue"
_QUEEN_COLOR  = "gold"
_SCOUT_COLOR  = "mediumpurple"

_FORAGER_COLORS = {
    ForagerState.RESTING:         "#b8860b",
    ForagerState.SCOUTING:        "darkorange",
    ForagerState.FLYING_TO_PATCH: "steelblue",
    ForagerState.COLLECTING:      "limegreen",
    ForagerState.RETURNING:       "crimson",
}

_EMPTY      = np.empty((0, 2))
_BASE_PAUSE = 0.04   # seconds per redraw at 1x speed


def _pts(lst: list) -> np.ndarray:
    return np.array(lst, dtype=float) if lst else _EMPTY


# ── Stateful renderers ────────────────────────────────────────────────────────

class _GridRenderer:
    """Creates all grid artists once; each frame only updates their data."""

    def __init__(self, ax, model):
        self.ax = ax
        ax.set_facecolor(_BG_COLOR)
        ax.set_xlim(0, model.width)
        ax.set_ylim(0, model.height)
        ax.set_aspect("equal")
        ax.set_title("Grid", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

        hx, hy = model.hive.pos
        self._hx, self._hy = hx, hy

        ax.add_patch(mpatches.FancyBboxPatch(
            (hx - 1.5, hy - 1.5), 3, 3,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor="black", facecolor=_HIVE_COLOR, zorder=2,
        ))
        ax.plot(hx + 0.5, hy + 0.9, "*", color=_QUEEN_COLOR,
                markersize=11, zorder=6, markeredgecolor="black", markeredgewidth=0.4)

        self._flower_circles = []
        for patch in model.flower_patches:
            px, py = patch.pos
            r = 0.15 + 0.45 * (patch.nectar / patch.max_nectar if patch.max_nectar else 0)
            circle = mpatches.Circle((px + 0.5, py + 0.5), r, color=_PATCH_COLOR, zorder=3)
            ax.add_patch(circle)
            self._flower_circles.append(circle)
        self._flower_patches = model.flower_patches

        self._nurse_sc = ax.scatter([], [], c=_NURSE_COLOR, s=12, zorder=5, linewidths=0)
        self._nurse_overflow = ax.text(
            hx + 0.5, hy - 1.3, "", color=_NURSE_COLOR,
            fontsize=6, ha="center", zorder=6, visible=False,
        )

        self._forager_sc = {
            state: ax.scatter([], [], c=color, s=20, zorder=4, linewidths=0)
            for state, color in _FORAGER_COLORS.items()
        }
        self._scout_sc = ax.scatter(
            [], [], c=_SCOUT_COLOR, marker="^", s=20, zorder=4, linewidths=0
        )
        self._count_texts: list = []
        self._arrows = None
        self._scout_excl: list = []

    def draw(self, model) -> None:
        for circle, patch in zip(self._flower_circles, self._flower_patches):
            r = 0.15 + 0.45 * (patch.nectar / patch.max_nectar if patch.max_nectar else 0)
            circle.set_radius(r)

        hx, hy = self._hx, self._hy
        nurses  = model.nurse_count
        visible = min(nurses, 12)
        nurse_pts = []
        for i in range(visible):
            row, col = divmod(i, 4)
            nurse_pts.append((hx - 1.1 + col * 0.75 + 0.5, hy - 0.6 + row * 0.65))
        self._nurse_sc.set_offsets(_pts(nurse_pts))
        if nurses > visible:
            self._nurse_overflow.set_text(f"+{nurses - visible}")
            self._nurse_overflow.set_visible(True)
        else:
            self._nurse_overflow.set_visible(False)

        state_pts: dict = {s: [] for s in _FORAGER_COLORS}
        scout_pts: list = []
        pos_counter: Counter = Counter()

        for agent in model.schedule.agents:
            if isinstance(agent, ForagerAgent) and agent.pos is not None:
                x, y = agent.pos
                state_pts[agent.state].append((x + 0.5, y + 0.5))
                pos_counter[agent.pos] += 1
            elif isinstance(agent, ScoutAgent) and agent.pos is not None:
                x, y = agent.pos
                scout_pts.append((x + 0.5, y + 0.5))
                pos_counter[agent.pos] += 1

        for state, sc in self._forager_sc.items():
            sc.set_offsets(_pts(state_pts[state]))
        self._scout_sc.set_offsets(_pts(scout_pts))

        needed = [(pos, cnt) for pos, cnt in pos_counter.items() if cnt > 1]
        for t in self._count_texts:
            t.set_visible(False)
        while len(self._count_texts) < len(needed):
            t = self.ax.text(
                0, 0, "", fontsize=5, ha="center", va="center",
                color="white", fontweight="bold", zorder=8, visible=False,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#111", alpha=0.7, linewidth=0),
            )
            self._count_texts.append(t)
        for i, (pos, cnt) in enumerate(needed):
            t = self._count_texts[i]
            t.set_position((pos[0] + 0.5, pos[1] + 0.5))
            t.set_text(str(cnt))
            t.set_visible(True)

        if self._arrows is not None:
            self._arrows.remove()
            self._arrows = None

        ax_x, ax_y, ax_u, ax_v, ax_c = [], [], [], [], []
        returning_scouts = []
        for agent in model.schedule.agents:
            if isinstance(agent, ForagerAgent) and agent.pos is not None:
                if agent.state == ForagerState.FLYING_TO_PATCH and agent.target_patch is not None:
                    target = agent.target_patch.pos
                elif agent.state == ForagerState.RETURNING:
                    target = model.hive.pos
                else:
                    target = None
                if target is not None:
                    x, y = agent.pos
                    dx, dy = target[0] - x, target[1] - y
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist == 0:
                        continue
                    ax_x.append(x + 0.5)
                    ax_y.append(y + 0.5)
                    ax_u.append(dx / dist * 0.6)
                    ax_v.append(dy / dist * 0.6)
                    ax_c.append(_FORAGER_COLORS[agent.state])
            elif (isinstance(agent, ScoutAgent) and agent.pos is not None
                    and agent.state == ScoutState.RETURNING):
                returning_scouts.append(agent.pos)

        for t in self._scout_excl:
            t.set_visible(False)
        while len(self._scout_excl) < len(returning_scouts):
            t = self.ax.text(
                0, 0, "!", fontsize=7, ha="center", va="bottom",
                color="red", fontweight="bold", zorder=9, visible=False,
            )
            self._scout_excl.append(t)
        for i, pos in enumerate(returning_scouts):
            t = self._scout_excl[i]
            t.set_position((pos[0] + 0.5, pos[1] + 0.7))
            t.set_visible(True)

        if ax_x:
            self._arrows = self.ax.quiver(
                ax_x, ax_y, ax_u, ax_v,
                color=ax_c, alpha=0.9, scale=1, scale_units="xy",
                angles="xy", width=0.002, headwidth=5, headlength=5, zorder=6,
            )


class _LegendRenderer:
    """Left panel: colony stats text at top, legend below with white title."""

    def __init__(self, ax):
        ax.set_facecolor("#1a1a2e")
        ax.set_axis_off()
        ax.text(0.5, 0.99, "Colony stats", transform=ax.transAxes,
                ha="center", va="top", color="white", fontsize=10, fontweight="bold")

        self._text = ax.text(
            0.08, 0.92, "", transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top", ha="left",
            fontfamily="monospace", color="white",
            bbox=dict(boxstyle="round", facecolor="#222", alpha=0.85, edgecolor="#555"),
        )

        leg = ax.legend(
            handles=_grid_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.0),
            fontsize=7, facecolor="#222", labelcolor="white",
            framealpha=0.85, title="Legend", title_fontsize=8,
        )
        leg.get_title().set_color("white")
        leg.get_frame().set_edgecolor("#555")

    def draw(self, model) -> None:
        self._text.set_text(
            f"Step     {model.schedule.steps:>6}\n\n"
            f"Nectar   {model.hive.nectar:>4.0f}/{MAX_NECTAR_STORES:.0f}\n"
            f"Brood    {model.hive.brood_count:>6}\n\n"
            f"Nurses   {model.nurse_count:>6}\n"
            f"Foragers {model.forager_count:>6}\n"
            f"Scouts   {model.scout_count:>6}\n"
        )


class _StatsRenderer:
    """Right panel: three separate trend charts (Nectar / Foragers / Nurses+Scouts)."""

    def __init__(self, ax):
        ax.set_facecolor("#1a1a2e")
        ax.set_axis_off()

        def _make_sub(bounds, title, title_color="white"):
            sub = ax.inset_axes(bounds)
            sub.set_facecolor("#0d0d1a")
            sub.set_title(title, color=title_color, fontsize=7, pad=2)
            sub.tick_params(colors="white", labelsize=5)
            for spine in sub.spines.values():
                spine.set_edgecolor("#555")
            return sub

        self._ax_nectar   = _make_sub([0.04, 0.69, 0.94, 0.27], "Nectar",          "gold")
        self._ax_foragers = _make_sub([0.04, 0.37, 0.94, 0.27], "Foragers",        "darkorange")
        self._ax_bees     = _make_sub([0.04, 0.05, 0.94, 0.27], "Nurses & Scouts", "white")

        self._ax_nectar.tick_params(labelbottom=False)
        self._ax_foragers.tick_params(labelbottom=False)

        self._ln_nectar,   = self._ax_nectar.plot([], [],   color="gold",        lw=1.5)
        self._ln_foragers, = self._ax_foragers.plot([], [], color="darkorange",   lw=1.2)

        self._ln_nurses, = self._ax_bees.plot([], [], color=_NURSE_COLOR, lw=1.2, label="Nurses")
        self._ln_scouts, = self._ax_bees.plot([], [], color=_SCOUT_COLOR, lw=1.2, label="Scouts")
        self._ax_bees.legend(fontsize=6, facecolor="#222", labelcolor="white", framealpha=0.8)

    def draw(self, model) -> None:
        data = model.datacollector.get_model_vars_dataframe()
        if len(data) < 2:
            return
        xs = np.arange(len(data))

        self._ln_nectar.set_data(xs,   data["Nectar"].values)
        self._ax_nectar.relim();   self._ax_nectar.autoscale_view()

        self._ln_foragers.set_data(xs, data["Foragers"].values)
        self._ax_foragers.relim(); self._ax_foragers.autoscale_view()

        self._ln_nurses.set_data(xs, data["Nurses"].values)
        self._ln_scouts.set_data(xs, data["Scouts"].values)
        self._ax_bees.relim();     self._ax_bees.autoscale_view()


# ── Public entry points ───────────────────────────────────────────────────────

def run_visualization(model, steps: int) -> None:
    """Live display. Space = pause/resume."""
    fig, (ax_left, ax_grid, ax_stats) = plt.subplots(
        1, 3, figsize=(15, 6.3),
        gridspec_kw={"width_ratios": [0.5, 2, 1]},
    )
    fig.patch.set_facecolor("#1a1a2e")
    fig.subplots_adjust(left=0.01, right=0.98, top=0.92, bottom=0.11, wspace=0.08)

    left_r  = _LegendRenderer(ax_left)
    grid_r  = _GridRenderer(ax_grid, model)
    stats_r = _StatsRenderer(ax_stats)

    ax_slider = fig.add_axes([0.18, 0.025, 0.36, 0.04], facecolor="#2a2a4a")
    speed_slider = mwidgets.Slider(
        ax_slider, "Speed", 0.05, 1.0, valinit=1.0, color="#4a90d9",
    )
    speed_slider.label.set_color("white")
    speed_slider.valtext.set_color("white")
    ax_slider.tick_params(colors="white")

    speed = [1.0]

    def _on_speed(val):
        speed[0] = val
        speed_slider.valtext.set_text(f"{val:.3f}x")

    speed_slider.on_changed(_on_speed)
    speed_slider.valtext.set_text("1.000x")

    paused = [False]

    def _on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]
            label = "PAUSED — Space to resume" if paused[0] else _title(model)
            fig.suptitle(label, color="white", fontsize=13)
            fig.canvas.draw_idle()

    _grid_xlim = ax_grid.get_xlim()
    _grid_ylim = ax_grid.get_ylim()

    def _on_scroll(event):
        if event.inaxes is not ax_grid or event.xdata is None:
            return
        factor = 0.85 if event.button == "up" else 1.0 / 0.85
        xc, yc = event.xdata, event.ydata
        xl, xr = ax_grid.get_xlim()
        yb, yt = ax_grid.get_ylim()
        new_xl = xc + (xl - xc) * factor
        new_xr = xc + (xr - xc) * factor
        new_yb = yc + (yb - yc) * factor
        new_yt = yc + (yt - yc) * factor
        if new_xr - new_xl > _grid_xlim[1] - _grid_xlim[0]:
            new_xl, new_xr = _grid_xlim
        if new_yt - new_yb > _grid_ylim[1] - _grid_ylim[0]:
            new_yb, new_yt = _grid_ylim
        ax_grid.set_xlim(new_xl, new_xr)
        ax_grid.set_ylim(new_yb, new_yt)
        fig.canvas.draw_idle()

    def _on_dblclick(event):
        if event.inaxes is ax_grid and event.dblclick:
            ax_grid.set_xlim(_grid_xlim)
            ax_grid.set_ylim(_grid_ylim)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event",        _on_scroll)
    fig.canvas.mpl_connect("button_press_event",  _on_dblclick)
    fig.canvas.mpl_connect("key_press_event",     _on_key)
    plt.ion()

    tick = range(steps) if steps is not None else itertools.count()
    for step in tick:
        while paused[0]:
            plt.pause(0.1)

        model.step()

        if step % VIZ_UPDATE_INTERVAL == 0:
            left_r.draw(model)
            grid_r.draw(model)
            stats_r.draw(model)
            fig.suptitle(_title(model), color="white", fontsize=13)
            remaining = max(0.005, _BASE_PAUSE / speed[0])
            while remaining > 0:
                plt.pause(min(0.05, remaining))
                remaining -= 0.05

    plt.ioff()
    plt.show()


def record_visualization(model, steps: int, output_path: str, fps: int = 30) -> None:
    """Headless recording to MP4. Requires: pip install imageio imageio-ffmpeg"""
    try:
        import imageio
    except ImportError:
        raise SystemExit("imageio not found — run: pip install imageio imageio-ffmpeg")

    steps = steps if steps is not None else DEFAULT_STEPS

    fig, (ax_left, ax_grid, ax_stats) = plt.subplots(
        1, 3, figsize=(15, 6),
        gridspec_kw={"width_ratios": [0.5, 2, 1]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#1a1a2e")

    left_r  = _LegendRenderer(ax_left)
    grid_r  = _GridRenderer(ax_grid, model)
    stats_r = _StatsRenderer(ax_stats)
    fig.canvas.draw()

    print(f"Recording {steps} steps → {output_path}  (fps={fps})")

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    try:
        for step in range(steps):
            model.step()
            left_r.draw(model)
            grid_r.draw(model)
            stats_r.draw(model)
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


def _grid_legend() -> list:
    return [
        mpatches.Patch(facecolor=_HIVE_COLOR,   edgecolor="k", label="Hive"),
        mpatches.Patch(facecolor=_PATCH_COLOR,               label="Flower patch"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=_QUEEN_COLOR,
                   markersize=10, label="Queen"),
        mpatches.Patch(facecolor=_NURSE_COLOR,               label="Nurse"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=_SCOUT_COLOR,
                   markersize=7, label="Scout"),
        mpatches.Patch(facecolor="darkorange",               label="Forager: scouting"),
        mpatches.Patch(facecolor="steelblue",                label="Forager: to patch"),
        mpatches.Patch(facecolor="limegreen",                label="Forager: collecting"),
        mpatches.Patch(facecolor="crimson",                  label="Forager: returning"),
        mpatches.Patch(facecolor="#b8860b",                  label="Forager: resting"),
    ]
