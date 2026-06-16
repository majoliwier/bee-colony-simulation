"""
Parameter sweep for the Bee Colony Simulation.

Sweeps three parameters, runs each combination with multiple seeds, and
saves results to results/sweep_<YYYYMMDD>.csv.

Usage (from repo root):
    python -m scripts.sweep
    python -m scripts.sweep --steps 300 --seeds 5
    python -m scripts.sweep --steps 200 --seeds 3 --out results/my_sweep.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics
from datetime import date
from itertools import product

from src.model import BeeColonyModel

# ── Sweep grid ────────────────────────────────────────────────────────────────
NURSE_AGES     = [30, 50, 60, 80]
TRAIL_DEPOSITS = [0.2, 0.4, 0.6]
PATCH_COUNTS   = [6, 12, 18, 24]

DEFAULT_STEPS = 300
DEFAULT_SEEDS = 3

FIELDS = [
    "nurse_to_forager_age", "trail_deposit_strength", "num_patches",
    "seed", "steps_run", "collapse_step", "final_nectar",
    "peak_foragers", "patches_found", "patches_total",
    "nurses_final", "foragers_final",
]


def run_one(nurse_age: int, trail: float, patches: int, steps: int, seed: int) -> dict:
    model = BeeColonyModel(
        num_patches=patches,
        seed=seed,
        nurse_to_forager_age=nurse_age,
        trail_deposit_strength=trail,
    )
    for _ in range(steps):
        model.step()
        if not model.running:
            break

    data = model.datacollector.get_model_vars_dataframe()
    peak_foragers = int(data["Foragers"].max()) if not data.empty else 0
    collapse_step = model.schedule.steps if not model.running else None

    return {
        "nurse_to_forager_age": nurse_age,
        "trail_deposit_strength": trail,
        "num_patches": patches,
        "seed": seed,
        "steps_run": model.schedule.steps,
        "collapse_step": collapse_step if collapse_step is not None else "",
        "final_nectar": round(model.hive.nectar, 2),
        "peak_foragers": peak_foragers,
        "patches_found": len(model.patch_discoveries),
        "patches_total": len(model.flower_patches),
        "nurses_final": model.nurse_count,
        "foragers_final": model.forager_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bee Colony parameter sweep")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Simulation steps per run (default {DEFAULT_STEPS})")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"Seeds per combo (default {DEFAULT_SEEDS})")
    parser.add_argument("--out",   default=None,
                        help="Output CSV path (default: results/sweep_YYYYMMDD.csv)")
    args = parser.parse_args()

    out_path = args.out or f"results/sweep_{date.today().strftime('%Y%m%d')}.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    combos = list(product(NURSE_AGES, TRAIL_DEPOSITS, PATCH_COUNTS))
    total  = len(combos) * args.seeds
    print(f"Sweep: {len(NURSE_AGES)} nurse ages × {len(TRAIL_DEPOSITS)} trail strengths × "
          f"{len(PATCH_COUNTS)} patch counts × {args.seeds} seeds = {total} runs")
    print(f"Steps per run: {args.steps}   Output: {out_path}\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        done = 0
        for nurse_age, trail, patches in combos:
            for seed in range(args.seeds):
                row = run_one(nurse_age, trail, patches, args.steps, seed)
                writer.writerow(row)
                f.flush()
                done += 1
                collapse = str(row["collapse_step"]) if row["collapse_step"] != "" else "-"
                print(
                    f"  [{done:>4}/{total}]  age={nurse_age:>2}  trail={trail:.1f}  "
                    f"patches={patches:>2}  seed={seed}  "
                    f"nectar={row['final_nectar']:>8.1f}  "
                    f"peak_for={row['peak_foragers']:>3}  collapse={collapse}",
                    flush=True,
                )

    print(f"\nSaved {done} rows → {out_path}")
    _print_summary(out_path)


def _print_summary(path: str) -> None:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    groups: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in rows:
        key = (r["nurse_to_forager_age"], r["trail_deposit_strength"], r["num_patches"])
        groups[key].append(float(r["final_nectar"]))

    print("\n-- Mean final nectar per parameter combo --")
    print(f"{'age':>4}  {'trail':>5}  {'patches':>7}  {'mean_nectar':>11}  {'std':>7}")
    print("-" * 46)
    for (age, trail, patches), vals in sorted(groups.items()):
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{age:>4}  {trail:>5}  {patches:>7}  {mean:>11.1f}  {std:>7.1f}")


if __name__ == "__main__":
    main()
