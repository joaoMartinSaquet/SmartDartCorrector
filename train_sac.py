"""train_sac_smartdart.py
================================
Train a Soft‑Actor Critic (SAC) agent on **SmartDartCorrector**’s custom Godot environment
using Stable‑Baselines3.

The script assumes you cloned https://github.com/joaoMartinSaquet/SmartDartCorrector and
installed its Python dependencies (including `godot-rl[sb3]`).

Default hyper‑parameters correspond to the off‑policy set you provided earlier, but can be
changed from the command line.

Example usage
-------------
```bash
pip install "stable-baselines3[extra]" godot-rl[sb3] tensorboard
python train_sac_smartdart.py --timesteps 1_000_000 --n-envs 4 --render
```
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from functools import partial
from typing import Any, Dict

# Make repo importable when this script lives outside the SmartDartCorrector root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.rolloutenv import smartDartEnv, VITE_USim  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    CheckpointCallback,
    ProgressBarCallback,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_smartdart_env(render: bool = False, perturbator: Any = None) -> smartDartEnv:  # type: ignore[name-defined]
    """Factory returning a fresh SmartDartEnv wrapped with Monitor."""
    # Initial position is placeholder – the env will override on reset
    u_sim = VITE_USim([0, 0])
    env = smartDartEnv(u_sim, perturbator, render=render, n_parallel=1)
    return Monitor(env)


def parse_policy_kwargs(text: str) -> Dict[str, Any]:
    """Safely parse a dict‑like string to actual dict (no `eval`)."""
    import ast
    value = ast.literal_eval(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("`policy_kwargs` must parse to a dict")
    return value

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Create vectorised environment
    env_fn = partial(make_smartdart_env, render=args.render)
    vec_env = DummyVecEnv([env_fn for _ in range(args.n_envs)])

    if args.normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # Instantiate SAC
    model = SAC(
        policy=args.policy,
        env=vec_env,
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
        verbose=1,
    )

    # 3️⃣  Callbacks
    ckpt_cb = CheckpointCallback(save_freq=20_000, save_path="models/sac", name_prefix="sac_sd")
    prog_cb = ProgressBarCallback()

    # 4️⃣  Train
    model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, prog_cb])

    # 5️⃣  Save final artefacts
    model.save("models/sac_smartdart_final")
    if isinstance(vec_env, VecNormalize):
        vec_env.save("models/vecnormalize.pkl")

    print("Training complete! Model and VecNormalize stats saved to ./models/")


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC on SmartDartCorrector’s custom environment (Godot RL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment params
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel SmartDart Godot instances")
    parser.add_argument("--render", action="store_true", help="Show Godot window(s)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False, help="Wrap with VecNormalize")

    # SAC hyper‑parameters (defaults = your provided set)
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total timesteps to train")
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

    # Policy setup
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument(
        "--policy-kwargs",
        type=parse_policy_kwargs,
        default="{'log_std_init': -3.67, 'net_arch': [64, 64]}",
        help="Dict passed to the policy constructor",
    )

    main(parser.parse_args())
