"""
Entry point: python -m src.run [--steps N] [--headless]
"""
import argparse
import statistics

from .config import DEFAULT_STEPS
from .model import BeeColonyModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Bee Colony Simulation")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Number of simulation steps (default: unlimited for GUI, {DEFAULT_STEPS} for --headless)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI and print final stats to stdout",
    )
    parser.add_argument(
        "--record",
        metavar="FILE",
        help="Record simulation to an MP4 file (e.g. --record sim.mp4). Requires ffmpeg.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for --record output (default: 30)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        metavar="N",
        help="Run N headless simulations with sequential seeds and print a comparison table",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (batch: seed for run 0, incremented per run)",
    )
    parser.add_argument(
        "--rl-foragers",
        action="store_true",
        help="Use PPO-controlled RL foragers instead of the default FSM foragers",
    )
    parser.add_argument(
        "--no-pheromones",
        action="store_true",
        help="Disable pheromone CA layer (pure random-walk scouting)",
    )
    args = parser.parse_args()

    if args.batch is not None:
        steps = args.steps if args.steps is not None else DEFAULT_STEPS
        _run_batch(args.batch, steps, args.seed, args.rl_foragers,
                   use_pheromones=not args.no_pheromones)
        return

    selected_pheromones = None
    selected_rl_foragers = args.rl_foragers
    if not args.headless and not args.record:
        from .visualization import choose_startup_mode

        selected_pheromones, selected_rl_foragers = choose_startup_mode()

    model = BeeColonyModel(seed=args.seed, use_rl_foragers=selected_rl_foragers)
    if args.no_pheromones:
        model.use_pheromones = False
    elif selected_pheromones is not None:
        model.use_pheromones = selected_pheromones

    if args.record:
        from .visualization import record_visualization

        record_visualization(model, args.steps, args.record, fps=args.fps)
        _print_stats(model)
    elif args.headless:
        steps = args.steps if args.steps is not None else DEFAULT_STEPS
        for _ in range(steps):
            model.step()
            if not model.running:
                break
        _print_stats(model)
    else:
        from .visualization import run_visualization

        run_visualization(model, args.steps, choose_mode=False)


def _print_stats(model: BeeColonyModel) -> None:
    print("\n-- Colony state ---------------------------------------")
    print(f"  Steps run :  {model.schedule.steps}")
    print(f"  Forager AI:  {'RL/PPO' if model.use_rl_foragers else 'FSM'}")
    print(f"  Nectar    :  {model.hive.nectar:.1f}")
    print(f"  Brood     :  {model.hive.brood_count}")
    print(f"  Nurses    :  {model.nurse_count}")
    print(f"  Foragers  :  {model.forager_count}")
    print(f"  Scouts    :  {model.scout_count}")
    print("------------------------------------------------------\n")

    discoveries = sorted(model.patch_discoveries, key=lambda d: d["step"])
    total = len(model.flower_patches)
    print(f"\n-- Patch discoveries ({len(discoveries)}/{total} found) --")

    dance_chains = {e["patch_pos"]: e for e in model.dance_log}

    if discoveries:
        for d in discoveries:
            print(
                f"\n  step {d['step']:>4}  [{d['finder']:>7}]  "
                f"pos={d['pos']}  quality={d['quality']:.2f}"
            )
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


def _run_batch(n_runs: int, steps: int, base_seed: int | None, use_rl_foragers: bool,
               use_pheromones: bool = True) -> None:
    rows = []
    for i in range(n_runs):
        seed = (base_seed + i) if base_seed is not None else i
        model = BeeColonyModel(seed=seed, use_rl_foragers=use_rl_foragers)
        model.use_pheromones = use_pheromones
        for _ in range(steps):
            model.step()
            if not model.running:
                break
        found = len(model.patch_discoveries)
        total = len(model.flower_patches)
        rows.append(
            {
                "run": i + 1,
                "seed": seed,
                "steps": model.schedule.steps,
                "nectar": round(model.hive.nectar, 1),
                "nurses": model.nurse_count,
                "foragers": model.forager_count,
                "scouts": model.scout_count,
                "patches_found": f"{found}/{total}",
                "_found_int": found,
            }
        )

    headers = ["Run", "Seed", "Steps", "Nectar", "Nurses", "Foragers", "Scouts", "Patches"]
    widths = [4, 6, 6, 8, 7, 9, 7, 10]
    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, widths))
    ai_label  = "RL/PPO" if use_rl_foragers else "FSM"
    ph_label  = "pheromones=ON" if use_pheromones else "pheromones=OFF"
    print(f"\n-- Batch results ({n_runs} runs x {steps} steps, foragers={ai_label}, {ph_label}) --")
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        vals = [
            row["run"],
            row["seed"],
            row["steps"],
            row["nectar"],
            row["nurses"],
            row["foragers"],
            row["scouts"],
            row["patches_found"],
        ]
        print(sep.join(str(v).ljust(w) for v, w in zip(vals, widths)))

    print()
    for key, label in [("nectar", "Nectar"), ("_found_int", "Patches found")]:
        nums = [r[key] for r in rows]
        mean = statistics.mean(nums)
        lo, hi = min(nums), max(nums)
        std_str = f"  std={statistics.stdev(nums):.2f}" if n_runs > 1 else ""
        print(f"  {label:<14} mean={mean:.1f}{std_str}  min={lo}  max={hi}")
    print()


if __name__ == "__main__":
    main()
