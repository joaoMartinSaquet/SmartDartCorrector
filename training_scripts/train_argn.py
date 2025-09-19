from pathlib import Path
import sys
import os

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
from agrn import  EATMuPlusLambda, gymProblem, GRN

from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
class grnSmartDart():

    def __init__(self, start_nreg, perturbator, reward_shaping, normalize):
        # super().__init__(env_info, env_name, max_ts)

        self.perturbator = perturbator
        self.reward_shaping = reward_shaping
        self.normalize = normalize
        self.nin = 2
        self.nout = 2
        self.nreg = start_nreg

        # self.env = smartDartEnv(VITE_USim(x_init=[0,0]), perturbator=self.perturbator, render=False, reward_shape=self.reward_shaping, normalize=self.normalize)




    def eval(self, genome):


        env = smartDartEnv(VITE_USim(x_init=[0,0]), perturbator=self.perturbator, render=False, reward_shape=self.reward_shaping, normalize=self.normalize)

        # print("environement created ", env)
        g = GRN(genome, self.nin, self.nout)
        g.setup()
        g.warmup(25)
        

        
        obs, info = env.reset()
        fit = 0
        done = False
        while not done:
            # print("step")
            g.set_input(obs)
            g.step(10)
            action = g.get_output() 
            action = action*80 - 40
            obs, reward, done, truncated, terminated, = env.step(action)

            fit += reward 
            if truncated or terminated: 
                done = True
            # print("step, done = ", done)
        env.close()
        # print("fit = ", fit)
        return fit, 

def main(args):
    


    reward_shaping = args.reward_shaping
    normalize = args.normalize
    ngen = args.ngen

    perturbator = None
    if args.perturbator == 'None':
        perturbator = None
    elif args.perturbator == 'RAM':
        perturbator = None
        logger.warning(f"Not implemented yet: {args.perturbator}")
    elif args.perturbator == 'Noise':
        perturbator = NormalJittering(10, 20)


    global envs
    envs = [smartDartEnv(VITE_USim([0, 0]), perturbator = perturbator, render = args.render, n_stack=args.nstack, normalize = args.normalize, reward_shape=True) for _ in range(args.n_envs)]
    # set_env_pool(envs)



    print("len envs ", len(envs))
    p = grnSmartDart(0, perturbator, reward_shaping, normalize)

    e = EATMuPlusLambda(nin = p.nin, nout = p.nout, nreg=p.nreg)

    alg, hist = e.run(100,p, 5, 5, multiproc=True, verbose=True)
    e.visualize_evolutions()

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        description="Train GRN on SmartDartCorrector or run Optuna tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment params
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--reward_shaping", action=argparse.BooleanOptionalAction, default=True)
    # CGP args
    parser.add_argument("--ngen", type=int, default=100)
    
    # smartDarts
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation during tuning")
    parser.add_argument("--perturbator", choices=['None', 'Noise'], default='None')
    parser.add_argument("--nstack", type=int, default=1)

    args = parser.parse_args()

    main(args)
