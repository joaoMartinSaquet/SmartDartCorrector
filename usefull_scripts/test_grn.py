from pathlib import Path
import sys
import os

import pandas as pd
from loguru import logger
import wandb
import pprint
from functools import partial
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Queue, Manager

# ---------------------------------------------------------------------------
# Repo‑local imports – SmartDart env
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from common.rolloutenv import *
from common.perturbation import *
from training_scripts.train_argn import grnSmartDart
# from GA.pyGRN.pygrn import problems, evolution, grns
from agrn import  EATMuPlusLambda, gymProblem, GRN
from agrn import   utils

import numpy as np




if __name__ == "__main__":

    genome_path = "/home/jmartinsaquet/Documents/SmartDartCorrector/models/GRN/genomes.csv"

    df = pd.read_csv(genome_path, sep=";")
    genome_test = eval(df["genome"][2])

    g = GRN(genome_test, 2, 4)

    grnSmartDart()

