from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from rl_forager_mvp.colony_env import BeeColonyForagerEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue PPO training in the real BeeColonyModel environment."
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patches", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--init-model", default="rl_forager_mvp/models/ppo_forager.zip")
    parser.add_argument("--out", default="rl_forager_mvp/models/ppo_forager_colony.zip")
    parser.add_argument("--best-out", default="rl_forager_mvp/models/ppo_forager_colony_best.zip")
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def make_env(args: argparse.Namespace, seed: int) -> Monitor:
    return Monitor(
        BeeColonyForagerEnv(
            max_steps=args.episode_steps,
            n_patches=args.patches,
            use_pheromones=True,
            seed=seed,
        )
    )


def run_baseline(policy_name: str, args: argparse.Namespace, episodes: int) -> tuple[float, float]:
    deposits, rewards = [], []
    for i in range(episodes):
        env = BeeColonyForagerEnv(
            max_steps=args.episode_steps,
            n_patches=args.patches,
            use_pheromones=True,
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
        deposits.append(env.deposited_total)
        rewards.append(total_reward)
    return float(np.mean(deposits)), float(np.mean(rewards))


def evaluate_deposit(model: PPO, args: argparse.Namespace, episodes: int) -> tuple[float, float]:
    deposits = []
    eval_env = make_env(args, args.seed + 20_000)
    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
        warn=False,
    )
    for i in range(episodes):
        env = BeeColonyForagerEnv(
            max_steps=args.episode_steps,
            n_patches=args.patches,
            use_pheromones=True,
            seed=args.seed + 20_000 + i,
        )
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        deposits.append(env.deposited_total)
    return float(np.mean(deposits)), float(mean_reward)


class DepositEvalCallback(BaseCallback):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.best_deposit = -np.inf
        self.best_path = Path(args.best_out)

    def _on_step(self) -> bool:
        if self.n_calls % self.args.eval_every != 0:
            return True

        mean_deposit, mean_reward = evaluate_deposit(
            self.model,
            self.args,
            self.args.eval_episodes,
        )
        print(
            f"eval step={self.num_timesteps:>7} "
            f"deposit={mean_deposit:7.2f} reward={mean_reward:8.2f}"
        )
        if mean_deposit > self.best_deposit:
            self.best_deposit = mean_deposit
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_path)
            print(f"new best colony model saved to {self.best_path}")
        return True


def main() -> None:
    args = parse_args()

    raw_env = BeeColonyForagerEnv(
        max_steps=args.episode_steps,
        n_patches=args.patches,
        use_pheromones=True,
        seed=args.seed,
    )
    if args.check_env:
        check_env(raw_env, warn=True)

    random_deposit, random_reward = run_baseline("random", args, args.eval_episodes)
    heuristic_deposit, heuristic_reward = run_baseline("heuristic", args, args.eval_episodes)
    print("Before colony training")
    print(f"random    deposit={random_deposit:7.2f} reward={random_reward:8.2f}")
    print(f"heuristic deposit={heuristic_deposit:7.2f} reward={heuristic_reward:8.2f}")

    env = make_env(args, args.seed)
    init_path = Path(args.init_model)
    if init_path.exists():
        model = PPO.load(init_path, env=env)
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.ent_coef = args.ent_coef
        print(f"Continuing from {init_path}")
    else:
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

    model.learn(total_timesteps=args.timesteps, callback=DepositEvalCallback(args))

    best_path = Path(args.best_out)
    if best_path.exists():
        model = PPO.load(best_path)
        print(f"Loaded best evaluated colony model from {best_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved deployable colony PPO model to {out_path}")

    rl_deposit, rl_reward = evaluate_deposit(model, args, args.eval_episodes)
    print("After colony training")
    print(f"rl        deposit={rl_deposit:7.2f} reward={rl_reward:8.2f}")
    print(f"best model path: {args.best_out}")


if __name__ == "__main__":
    main()
