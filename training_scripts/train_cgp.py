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
        description="Train SAC on SmartDartCorrector or run Optuna tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment params
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)

    # SAC defaults (used when Optuna disabled)
    parser.add_argument("--ngen", type=int, default=10)
    
    # Optuna knobs
    parser.add_argument("--optuna-trials", type=int, default=0, help=">0 to enable tuning")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation during tuning")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

