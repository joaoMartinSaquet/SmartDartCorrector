from pathlib import Path
import sys

import pandas as pd
from loguru import logger
import wandb
import pprint
from functools import partial
import argparse
# ---------------------------------------------------------------------------
# Repo‑local imports – SmartDart env
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from common.rolloutenv import *
from common.perturbation import *
from classic_rl.rl_corrector import *
from GA.cgp_corrector import *




def main(args):

    parser = argparse.ArgumentParser(
        description="Train CGP on SmartDartCorrector or run Optuna tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment params
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)

    # CGP args
    parser.add_argument("--ngen", type=int, default=10)
    parser.add_argument("--offsprings", type=int, default=96)
    parser.add_argument("--parents", type=int, default=8)
    
    parser.add_argument("--col", type=int, default=10)
    parser.add_argument("--row", type=int, default=1)
    
    # smartDarts
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation during tuning")
    parser.add_argument("--perturbator", choices=['None', 'Noise'], default='None')
    parser.add_argument("--nstack", type=int, default=1)

    args = parser.parse_args()
    
    perturbator = None
    if args.perturbator == 'None':
        perturbator = None
    elif logger.info.perturbator == 'RAM':
        perturbator = None
        logger.warning(f"Not implemented yet: {args.perturbator}")
    elif args.perturbator == 'Noise':
        perturbator = NormalJittering(0, 20)

    n_stack = args.nstack
    if n_stack <= 0:
        logger.warning("n_stack must be greater than 0")
    

    logger.info("HYPERPARAMETERS")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")


    env = smartDartEnv(u_sim = VITE_USim([0, 0]), perturbator = perturbator, render = args.render, n_stack=n_stack, normalize = args.normalize, reward_shape=True)

    corrector = CGPCorrector(env, args.ngen, args.eval_episodes, args.col, args.row, perturbator = perturbator)

    corrector = corrector.learn()