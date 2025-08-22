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
# from GA.pyGRN.pygrn import problems, evolution, grns
from pygrn import problems, evolution, grns
from pygrn.problems.reinforcement import ReinforcementLearningTask, set_env_pool, create_env_info

def main(args):


    
    num_worker = args.num_worker

    perturbator = None
    if args.perturbator == 'None':
        perturbator = None
    elif logger.info.perturbator == 'RAM':
        perturbator = None
        logger.warning(f"Not implemented yet: {args.perturbator}")
    elif args.perturbator == 'Noise':
        perturbator = NormalJittering(10, 20)

    n_stack = 1
    

    logger.info("HYPERPARAMETERS")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    
    # create the env poolof size num_worker
    envs = [smartDartEnv(VITE_USim([0, 0]), perturbator = perturbator, render = args.render, n_stack=n_stack, normalize = args.normalize, reward_shape=True) for _ in range(num_worker)]
    set_env_pool(envs)
    
    problem = ReinforcementLearningTask(env_info=create_env_info(envs[0]), env_name="SmartDartCorrector")
    

    grn = lambda : grns.ClassicGRN()
    grneat = evolution.Evolution(problem, grn, num_workers=num_worker)
    best_fit, best_ind = grneat.run(args.ngen)

    # evaluate the best fit on the test set
    test_env = smartDartEnv(VITE_USim([0, 0]), perturbator = perturbator, render = args.render, n_stack=n_stack, normalize = args.normalize, reward_shape=False)
    test_prob = ReinforcementLearningTask(env = test_env, env_info=create_env_info(test_env), env_name="SmartDartCorrector")

    test_fit  = test_prob.eval(best_ind.grn)

    logger.info(f"Best fit: {best_fit}")
    logger.info(f"Test fit: {test_fit}")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        description="Train GRN on SmartDartCorrector or run Optuna tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment params
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_worker", type=int, default=10)
    parser.add_argument("--reward_shaping", action=argparse.BooleanOptionalAction, default=True)
    # CGP args
    parser.add_argument("--ngen", type=int, default=100)
    
    # smartDarts
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation during tuning")
    parser.add_argument("--perturbator", choices=['None', 'Noise'], default='None')
    parser.add_argument("--nstack", type=int, default=1)

    args = parser.parse_args()

    main(args)
