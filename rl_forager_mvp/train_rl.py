from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from rl_forager_mvp.env import BeeForagerEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for a minimal bee forager.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--patches", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--out", default="rl_forager_mvp/models/ppo_forager.zip")
    parser.add_argument("--best-out", default="rl_forager_mvp/models/ppo_forager_best.zip")
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def make_env(args: argparse.Namespace, seed: int) -> Monitor:
    return Monitor(
        BeeForagerEnv(
            grid_size=args.grid_size,
            n_patches=args.patches,
            max_steps=args.episode_steps,
            seed=seed,
        )
    )


def run_baseline(policy_name: str, args: argparse.Namespace, episodes: int) -> tuple[float, float]:
    nectar, rewards = [], []
    for i in range(episodes):
        env = BeeForagerEnv(
            grid_size=args.grid_size,
            n_patches=args.patches,
            max_steps=args.episode_steps,
            seed=args.seed + 10_000 + i,
        )
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(env.max_steps):
            if policy_name == "heuristic":
                action = env.heuristic_action()
            elif policy_name == "random":
                action = int(env.action_space.sample())
            else:
                raise ValueError(f"Unknown baseline: {policy_name}")
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        nectar.append(env.hive_nectar)
        rewards.append(total_reward)
    return float(np.mean(nectar)), float(np.mean(rewards))


def evaluate_nectar(model: PPO, args: argparse.Namespace, episodes: int) -> tuple[float, float]:
    nectar = []
    eval_env = make_env(args, args.seed + 20_000)
    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
        warn=False,
    )
    for i in range(episodes):
        env = BeeForagerEnv(
            grid_size=args.grid_size,
            n_patches=args.patches,
            max_steps=args.episode_steps,
            seed=args.seed + 20_000 + i,
        )
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        nectar.append(env.hive_nectar)
    return float(np.mean(nectar)), float(mean_reward)


class NectarEvalCallback(BaseCallback):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.best_nectar = -np.inf
        self.best_path = Path(args.best_out)

    def _on_step(self) -> bool:
        if self.n_calls % self.args.eval_every != 0:
            return True

        mean_nectar, mean_reward = evaluate_nectar(
            self.model,
            self.args,
            self.args.eval_episodes,
        )
        print(
            f"eval step={self.num_timesteps:>7} "
            f"nectar={mean_nectar:7.2f} reward={mean_reward:8.2f}"
        )
        if mean_nectar > self.best_nectar:
            self.best_nectar = mean_nectar
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_path)
            print(f"new best model saved to {self.best_path}")
        return True


def main() -> None:
    args = parse_args()

    raw_env = BeeForagerEnv(
        grid_size=args.grid_size,
        n_patches=args.patches,
        max_steps=args.episode_steps,
        seed=args.seed,
    )
    if args.check_env:
        check_env(raw_env, warn=True)

    env = make_env(args, args.seed)
    random_nectar, random_reward = run_baseline("random", args, args.eval_episodes)
    heuristic_nectar, heuristic_reward = run_baseline("heuristic", args, args.eval_episodes)
    print("Before training")
    print(f"random    nectar={random_nectar:7.2f} reward={random_reward:8.2f}")
    print(f"heuristic nectar={heuristic_nectar:7.2f} reward={heuristic_reward:8.2f}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
    )
    model.learn(total_timesteps=args.timesteps, callback=NectarEvalCallback(args))

    best_path = Path(args.best_out)
    if best_path.exists():
        model = PPO.load(best_path)
        print(f"Loaded best evaluated model from {best_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved deployable PPO model to {out_path}")

    rl_nectar, rl_reward = evaluate_nectar(model, args, args.eval_episodes)
    print("After training")
    print(f"rl        nectar={rl_nectar:7.2f} reward={rl_reward:8.2f}")
    print(f"best model path: {args.best_out}")


if __name__ == "__main__":
    main()
