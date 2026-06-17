from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

from rl_forager_mvp.env import BeeForagerEnv


@dataclass
class EpisodeResult:
    reward: float
    hive_nectar: float
    steps: int
    terminated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL forager.")
    parser.add_argument("--model", default="rl_forager_mvp/models/ppo_forager_colony.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--patches", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--render-one", action="store_true")
    return parser.parse_args()


def run_policy(policy_name: str, env: BeeForagerEnv, model: PPO | None = None) -> EpisodeResult:
    obs, _ = env.reset()
    total_reward = 0.0
    terminated = False

    for _ in range(env.max_steps):
        if policy_name == "rl":
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif policy_name == "heuristic":
            action = env.heuristic_action()
        elif policy_name == "random":
            action = int(env.action_space.sample())
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return EpisodeResult(
        reward=total_reward,
        hive_nectar=env.hive_nectar,
        steps=env.steps,
        terminated=terminated,
    )


def summarize(name: str, rows: list[EpisodeResult]) -> None:
    rewards = np.array([r.reward for r in rows], dtype=float)
    nectar = np.array([r.hive_nectar for r in rows], dtype=float)
    deaths = sum(r.terminated for r in rows)
    print(
        f"{name:<10} | "
        f"nectar mean={nectar.mean():7.2f} std={nectar.std():6.2f} | "
        f"reward mean={rewards.mean():8.2f} | "
        f"deaths={deaths:>2}/{len(rows)}"
    )


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model)

    results = {"random": [], "heuristic": [], "rl": []}
    for i in range(args.episodes):
        seed = args.seed + i
        for policy_name in results:
            env = BeeForagerEnv(
                grid_size=args.grid_size,
                n_patches=args.patches,
                max_steps=args.episode_steps,
                seed=seed,
            )
            results[policy_name].append(
                run_policy(policy_name, env, model if policy_name == "rl" else None)
            )

    print("\nPolicy comparison")
    print("-----------------")
    summarize("random", results["random"])
    summarize("heuristic", results["heuristic"])
    summarize("rl", results["rl"])

    if args.render_one:
        env = BeeForagerEnv(
            grid_size=args.grid_size,
            n_patches=args.patches,
            max_steps=args.episode_steps,
            seed=args.seed,
        )
        obs, _ = env.reset()
        for _ in range(40):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        print("\nExample final grid after 40 RL steps")
        print("------------------------------------")
        print(env.render())


if __name__ == "__main__":
    main()

