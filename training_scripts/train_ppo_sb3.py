#!/usr/bin/env python3
"""train_sac_smartdart.py
====================================
Train **SAC** on SmartDartCorrector’s custom Godot environment with optional
**Optuna** hyper‑parameter optimisation.

Usage examples
--------------
# Plain training run (no tuning)
python train_sac_smartdart.py --timesteps 100_000 --n-envs 2

# Run Optuna for 50 trials, each 50k steps, evaluate on 5 episodes
python train_sac_smartdart.py --timesteps 50_000 \
    --optuna-trials 50 --eval-episodes 5 --n-envs 1

All Optuna‑selected parameters are printed at the end; you can retrain the
best model by passing them back via CLI flags.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict
import loguru

# ---------------------------------------------------------------------------
# Third‑party deps (installed separately)
# ---------------------------------------------------------------------------
import optuna
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

# ---------------------------------------------------------------------------
# Repo‑local imports – SmartDart env
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.rolloutenv import smartDartEnv, VITE_USim # noqa: E402  (after sys.path tweak)
from common.perturbation import *

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_smartdart_env(render: bool = False, perturbator: Any | None = None, nstack: int = 1):
    """Factory for a fresh SmartDartEnv wrapped with Monitor."""
    u_sim = VITE_USim([0, 0])
    env = smartDartEnv(u_sim, perturbator, render=render, n_parallel=1, n_stack=nstack)
    return Monitor(env)


def parse_policy_kwargs(text: str) -> Dict[str, Any]:
    import ast

    value = ast.literal_eval(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("`policy_kwargs` must parse to a dict")
    return value


# ---------------------------------------------------------------------------
# Optuna utilities
# ---------------------------------------------------------------------------

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Parameter search space for PPO to check."""

    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000, 200_000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 256, 512, 1024]),
        "tau": trial.suggest_float("tau", 0.005, 0.02),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "train_freq": trial.suggest_categorical("train_freq", [16, 32, 64]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [16, 32, 64]),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.0, 0.01, 0.1]),
        "use_sde": trial.suggest_categorical("use_sde", [True, False]),
    }


# ---------------------------------------------------------------------------
# Main training / tuning routines
# ---------------------------------------------------------------------------

def make_vec_envs(perturbator : Any | None, n_envs: int, render: bool, normalize: bool):
    env_fn = partial(make_smartdart_env, render=render, perturbator=perturbator)
    venv = DummyVecEnv([env_fn for _ in range(n_envs)])
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
    return venv


def train_once(args: argparse.Namespace, **model_kwargs) -> tuple[SAC, float]:
    """Train SAC once and return (model, mean_reward)."""

    vec_env = make_vec_envs(args.n_envs, args.render, args.normalize)

    model = PPO(
        policy=args.policy,
        env=vec_env,
        learning_starts=args.learning_starts,
        policy_kwargs=args.policy_kwargs,
        verbose=0,
        **model_kwargs,
    )
    print("Training model for {} timesteps".format(args.timesteps))
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=args.eval_episodes)

    return model, mean_reward


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    params = sample_ppo_params(trial)
    model, mean_reward = train_once(args, **params)

    # Track best model path for later retrieval
    trial.set_user_attr("mean_reward", mean_reward)
    return mean_reward  # maximise


def run_optuna(args: argparse.Namespace):
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: objective(t, args), n_trials=args.optuna_trials, show_progress_bar=True)

    print("\n Best trial (#{}) — reward {:.2f}".format(study.best_trial.number, study.best_value))
    print("Best hyper‑parameters:\n" + "\n".join(f"  {k}: {v}" for k, v in study.best_trial.params.items()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train SAC on SmartDartCorrector or run Optuna tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment params
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)

    # SAC defaults (used when Optuna disabled)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.9999)
    parser.add_argument("--train-freq", type=int, default=32)
    parser.add_argument("--gradient-steps", type=int, default=32)
    parser.add_argument("--ent-coef", type=float, default=0.1)
    parser.add_argument("--use-sde", action=argparse.BooleanOptionalAction, default=True)

    # Policy
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--policy-kwargs", type=parse_policy_kwargs,
                        default="{'log_std_init': -3.67, 'net_arch': [64, 64]}")

    # Optuna knobs
    parser.add_argument("--optuna-trials", type=int, default=0, help=">0 to enable tuning")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation during tuning")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.optuna_trials > 0:
        run_optuna(args)
    else:
        # Single run with CLI hyper‑parameters
        model_kwargs = {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "ent_coef": args.ent_coef,
            "use_sde": args.use_sde,
        }



        perturbator = None
        if args.perturbator == 'None':
            perturbator = None
        elif logger.info.perturbator == 'RAM':
            perturbator = None
            logger.warning(f"Not implemented yet: {args.perturbator}")
        elif args.perturbator == 'Noise':
            perturbator = NormalJittering(0, 20)

        vec_env = make_vec_envs(None, args.n_envs, args.render, args.normalize)

        model = PPO(
            policy=args.policy,
            env=vec_env,
            learning_starts=args.learning_starts,
            policy_kwargs=args.policy_kwargs,
            verbose=1,
            **model_kwargs,
            tensorboard_log="./logs/tensorboard",
        )

        # Callbacks / saving (only in non‑Optuna mode)
        ckpt_cb = CheckpointCallback(save_freq=5000, save_path="models/sac", name_prefix="sac_sd")
        prog_cb = ProgressBarCallback()
        # Re‑learn with callbacks so progress shows
        model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, prog_cb])
        model.save("models/sac_smartdart_final")
        # print(f"Finished training — mean eval reward: {mean_reward:.2f}")


if __name__ == "__main__":
    main()
