#!/usr/bin/env python3
"""
Train a Soft Actor‑Critic (SAC) agent with Stable‑Baselines3.

Default hyper‑parameters (can be overridden via CLI):
OrderedDict([
    ('batch_size', 512),
    ('buffer_size', 50_000),
    ('ent_coef', 0.1),
    ('gamma', 0.9999),
    ('gradient_steps', 32),
    ('learning_rate', 3e-4),
    ('learning_starts', 0),
    ('n_timesteps', 50_000),
    ('policy', 'MlpPolicy'),
    ('policy_kwargs', {'log_std_init': -3.67, 'net_arch': [64, 64]}),
    ('tau', 0.01),
    ('train_freq', 32),
    ('use_sde', True),
    ('normalize', False),
])

Example
-------
$ python train_sac_sb3.py --env-id Pendulum-v1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym  # noqa: F401  # envs may be created via gym.make
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
    BaseCallback
)

# callback

# Dictionary to store logs
logs = {
    'timestep': [],
    'reward': [],
    'episode': []
}

class LogCallback(BaseCallback):
    def __init__(self, log_interval=1000, verbose=0):
        super(LogCallback, self).__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        # Log every log_interval steps
        if self.num_timesteps % self.log_interval == 0:
            # Get the mean reward from the training instance
            mean_reward = self.locals.get('rewards', None)
            ep = self.locals.get('episode', None)
            if mean_reward is not None and ep is not None and ep > 0:
                mean_reward = sum(mean_reward) / ep

                # Log the mean reward and steps
                logs['timestep'].append(self.num_timesteps)
                logs['reward'].append(mean_reward) 


                if self.verbose > 0:
                    print(f"Step: {self.num_timesteps}, Mean Reward: {mean_reward}")

        return True
# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _str_to_dict(txt: str | dict | None) -> dict[str, Any]:
    """Safely convert a CLI string representing a dict into an actual dict."""
    if txt is None:
        return {}
    if isinstance(txt, dict):
        return txt
    try:
        # *Very* small surface area – only literal eval of dicts.
        from ast import literal_eval

        result = literal_eval(txt)
        if not isinstance(result, dict):
            raise ValueError
        return result
    except Exception as exc:  # noqa: BLE001
        pass


def make_env(env_id: str, n_envs: int, normalize: bool):
    """Create (optionally normalised) vectorised training environment."""
    env = make_vec_env(env_id, n_envs=n_envs, monitor_dir="./logs/monitor")
    env = VecMonitor(env)
    if normalize:
        env = VecNormalize(env)
    return env


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    # 1. Environment
    env = make_env(args.env_id, args.n_envs, args.normalize)

    # 2. SAC model
    model = SAC(
        policy=args.policy,
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        use_sde=args.use_sde,
        policy_kwargs=args.policy_kwargs,
        tensorboard_log="./logs/tensorboard",
        verbose=1,
    )

    # 3. Callbacks (checkpoints, evaluation, progress bar)
    save_dir = Path("models") / "sac"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(save_freq=5_000, save_path=save_dir, name_prefix="sac")

    eval_env = make_env(args.env_id, 1, args.normalize)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    progress_cb = ProgressBarCallback()
    log_cb = LogCallback(log_interval=1000)
    # 4. Learn
    model.learn(
        total_timesteps=args.n_timesteps,
        callback=[checkpoint_cb, eval_cb, log_cb],
    )

    # 5. Save artefacts
    model.save(save_dir / "sac_final")
    if isinstance(env, VecNormalize):
        env.save(save_dir / "vecnormalize.pkl")

    print("Training complete! Model and VecNormalize statistics are stored in", save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC with Stable‑Baselines3 using bespoke hyper‑parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-id", default="MountainCarContinuous-v0", help="Gymnasium environment ID")
    # Environment params
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)

    # SAC hyper‑parameters
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

    # Policy settings
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument(
        "--policy-kwargs",
        type=_str_to_dict,
        default="{'log_std_init': -3.67, 'net_arch': [64, 64]}",
        help="Dictionary of keyword arguments for the policy architecture.",
    )

    # Training loop params
    parser.add_argument("--n-timesteps", type=int, default=50_000)

    args = parser.parse_args()
    main(args)
