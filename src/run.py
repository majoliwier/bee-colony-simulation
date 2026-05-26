"""
Entry point — run as:  python -m src.run  [--steps N] [--headless]
"""
import argparse

from .model import BeeColonyModel
from .config import DEFAULT_STEPS


def main() -> None:
    parser = argparse.ArgumentParser(description="Bee Colony Simulation")
    parser.add_argument(
        "--steps", type=int, default=None,
        help=f"Number of simulation steps (default: unlimited for GUI, {DEFAULT_STEPS} for --headless)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI and print final stats to stdout",
    )
    parser.add_argument(
        "--record", metavar="FILE",
        help="Record simulation to an MP4 file (e.g. --record sim.mp4). Requires ffmpeg.",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for --record output (default: 30)",
    )
    args = parser.parse_args()

    model = BeeColonyModel()

    if args.record:
        from .visualization import record_visualization
        record_visualization(model, args.steps, args.record, fps=args.fps)
    elif args.headless:
        steps = args.steps if args.steps is not None else DEFAULT_STEPS
        for _ in range(steps):
            model.step()
        _print_stats(model)
    else:
        from .visualization import run_visualization
        run_visualization(model, args.steps)  # None → run until window closed


def _print_stats(model: BeeColonyModel) -> None:
    from .agents.nurse import NurseAgent
    from .agents.forager import ForagerAgent

    nurses   = sum(1 for a in model.schedule.agents if isinstance(a, NurseAgent))
    foragers = sum(1 for a in model.schedule.agents if isinstance(a, ForagerAgent))

    print("\n── Colony state ──────────────────────────────────────")
    print(f"  Steps run :  {model.schedule.steps}")
    print(f"  Nectar    :  {model.hive.nectar:.1f}")
    print(f"  Brood     :  {model.hive.brood_count}")
    print(f"  Nurses    :  {nurses}")
    print(f"  Foragers  :  {foragers}")
    print("──────────────────────────────────────────────────────\n")

    data = model.datacollector.get_model_vars_dataframe()
    if not data.empty:
        print(data.tail(10).to_string())
        print()


if __name__ == "__main__":
    main()
