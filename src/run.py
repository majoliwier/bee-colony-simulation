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
        _print_stats(model)
    elif args.headless:
        steps = args.steps if args.steps is not None else DEFAULT_STEPS
        for _ in range(steps):
            model.step()
        _print_stats(model)
    else:
        from .visualization import run_visualization
        run_visualization(model, args.steps)  # None → run until window closed


def _print_stats(model: BeeColonyModel) -> None:
    print("\n── Colony state ──────────────────────────────────────")
    print(f"  Steps run :  {model.schedule.steps}")
    print(f"  Nectar    :  {model.hive.nectar:.1f}")
    print(f"  Brood     :  {model.hive.brood_count}")
    print(f"  Nurses    :  {model.nurse_count}")
    print(f"  Foragers  :  {model.forager_count}")
    print(f"  Scouts    :  {model.scout_count}")
    print("──────────────────────────────────────────────────────\n")

    discoveries = sorted(model.patch_discoveries, key=lambda d: d["step"])
    total = len(model.flower_patches)
    print(f"\n── Patch discoveries ({len(discoveries)}/{total} found) ──")

    # build lookup: patch_pos -> first dance log entry (for attaching recruit chains)
    dance_chains = {e["patch_pos"]: e for e in model.dance_log}

    if discoveries:
        for d in discoveries:
            print(f"\n  step {d['step']:>4}  [{d['finder']:>7}]  pos={d['pos']}  quality={d['quality']:.2f}")
            chain = dance_chains.get(d["pos"])
            if chain and chain["recruits"]:
                for r in chain["recruits"]:
                    if r["arrived_step"] is not None:
                        status = f"arrived step {r['arrived_step']:>4}"
                    else:
                        status = "never arrived (died or patch depleted)"
                    print(f"    +- forager #{r['forager_id']:<3}  {status}")
            else:
                print("    (no resting foragers when dance performed)")
    else:
        print("  none")
    print()

    data = model.datacollector.get_model_vars_dataframe()
    if not data.empty:
        print(data.tail(10).to_string())
        print()


if __name__ == "__main__":
    main()
