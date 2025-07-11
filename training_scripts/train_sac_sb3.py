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
from loguru import logger
import time

# ---------------------------------------------------------------------------
# Third‑party deps (installed separately)
# ---------------------------------------------------------------------------
import optuna
import optuna.visualization as vis 
from optuna.samplers import TPESampler
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.logger import configure
import torch as th


# ---------------------------------------------------------------------------
# Repo‑local imports – SmartDart env
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.rolloutenv import smartDartEnv, VITE_USim  # noqa: E402  (after sys.path tweak)
from common.perturbation import *

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_smartdart_env(perturbator , render: bool = False, nstack: int = 1, normalize: bool = False):
    """Factory for a fresh SmartDartEnv wrapped with Monitor."""
    u_sim = VITE_USim([0, 0])
    env = smartDartEnv(u_sim, perturbator, render=render, n_parallel=1, n_stack=nstack, normalize=normalize)
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

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Parameter search space for SAC."""

    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000, 200_000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 256, 512, 1024]),
        "tau": trial.suggest_float("tau", 0.005, 0.02),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "train_freq": trial.suggest_categorical("train_freq", [100, 1000, 4000, 5000]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [16, 32, 64]),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.0, 0.01, 0.1]),
        "use_sde": trial.suggest_categorical("use_sde", [True, False]),
    }


# ---------------------------------------------------------------------------
# Main training / tuning routines
# ---------------------------------------------------------------------------

def make_vec_envs(perturbator, n_envs: int, render: bool, normalize: bool, nstack: int = 1):
    env_fn = partial(make_smartdart_env,perturbator, render=render, nstack=nstack, normalize=normalize)
    venv = DummyVecEnv([env_fn for _ in range(n_envs)])
    return venv


def train_once(args: argparse.Namespace,nstack = 1, **model_kwargs) -> tuple[SAC, float]:
    """Train SAC once and return (model, mean_reward)."""

    vec_env = make_vec_envs(args.n_envs, args.render, args.normalize, nstack=nstack)

    model = SAC(
        policy=args.policy,
        env=vec_env,
        learning_starts=args.learning_starts,
        verbose=0,
        **model_kwargs,
    )
    print("Training model for {} timesteps".format(args.timesteps))
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=args.eval_episodes)

    return model, mean_reward


def objective(trial: optuna.Trial, args: argparse.Namespace, n_stack = 1) -> float:
    params = sample_sac_params(trial)
    model, mean_reward = train_once(args, **params, nstack=n_stack)

    # Track best model path for later retrieval
    trial.set_user_attr("mean_reward", mean_reward)
    return mean_reward  # maximise


def run_optuna(args: argparse.Namespace, nstack = 1):
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: objective(t, args, nstack), n_trials=args.optuna_trials, show_progress_bar=True)

    print("\n Best trial (#{}) — reward {:.2f}".format(study.best_trial.number, study.best_value))
    print("Best hyper‑parameters:\n" + "\n".join(f"  {k}: {v}" for k, v in study.best_trial.params.items()))

    # Generate and show the optimization history plot
    opt_history_fig = vis.plot_optimization_history(study)
    opt_history_fig.show()

    # Generate and show the parallel coordinate plot
    parallel_fig = vis.plot_parallel_coordinate(study)
    parallel_fig.show()

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
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)

    # SAC defaults (used when Optuna disabled)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--learning-rate", type=float, default=5.95e-5)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=100)
    parser.add_argument("--gradient-steps", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01   )
    parser.add_argument("--use-sde", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perturbator", choices=['None', 'Noise'], default='None')
    
    # Policy
    parser.add_argument("--policy", default="MlpPolicy")

    # Optuna knobs
    parser.add_argument("--optuna-trials", type=int, default=0, help=">0 to enable tuning")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation during tuning")
    parser.add_argument("--seed", type=int, default=42)
    n_stack = 5
    args = parser.parse_args()


    logger.info("HYPERPARAMETERS")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")



    # Custom network architecture
    policy_kwargs = dict(
    net_arch=[256, 256],  # Two hidden layers of 256 units each
    activation_fn=th.nn.Tanh,  # Use ReLU activation function
    )

    perturbator = None
    if args.perturbator == 'None':
        perturbator = None
    elif args.perturbator == 'RAM':
        perturbator = None
        logger.warning(f"Not implemented yet: {args.perturbator}")
    elif args.perturbator == 'Noise':
        perturbator = NormalJittering(0, 20)


    # # used args
    # print("args received :")
    # for i, arg in enumerate(args, start=1):
    #     print(f"Arg {i}: {arg}")

    if args.optuna_trials > 0:
        run_optuna(args, nstack=n_stack)
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

        
        vec_env = make_vec_envs(perturbator, args.n_envs, args.render, args.normalize, nstack=n_stack)

        # create logger
        path = f'logs_corrector/SAC_{time.strftime("%Y%m%d-%H%M%S")}/'
        sb3_logger = configure(path, ["stdout", "csv", "tensorboard"])


        model = SAC(
            policy=args.policy,
            env=vec_env,
            learning_starts=args.learning_starts,
            verbose=1,
            **model_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/tensorboard",
        )
        model.set_logger(sb3_logger)

        # Callbacks / saving (only in non‑Optuna mode)
        ckpt_cb = CheckpointCallback(save_freq=5000, save_path="models/sac", name_prefix="sac_sd")
        prog_cb = ProgressBarCallback()
        # Re‑learn with callbacks so progress shows
        model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, prog_cb])
        model.save("models/sac_smartdart_final")


        # evaluation 
        n_ep = 10
        env = smartDartEnv(VITE_USim([0, 0]), None, render=False, n_parallel=1, n_stack=n_stack, normalize=True, reward_shape=False)
        total_reward = []
        for i in range(n_ep):

            obs, _ = env.reset()

            # Rollout
            reward_ep = 0
            done = False
            while not done:
                # Use deterministic policy (no exploration noise)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                reward_ep += reward
                # env.render()  # optional

            total_reward.append(reward_ep)
            print(f"Episode {i} reward: {reward_ep:.2f}")

        print(f"Episode {i} reward: {sum(total_reward)/n_ep:.2f}")
            
if __name__ == "__main__":
    main()
