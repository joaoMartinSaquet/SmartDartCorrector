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
# from GA.pyGRN.pygrn import problems, evolution, grns
from agrn import  EATMuPlusLambda, gymProblem, GRN
import socket

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))             # let OS pick a free port
    port = s.getsockname()[1]   # get the port number
    s.close()
    return port

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

    def __init__(self, n_envs, start_nreg, perturbator, reward_shaping, normalize):
        # super().__init__(env_info, env_name, max_ts)

        self.perturbator = perturbator
        self.reward_shaping = reward_shaping
        self.normalize = normalize
        self.nin = 2
        self.nout = 2
        self.nreg = start_nreg
    

        # self.env = smartDartEnv(VITE_USim(x_init=[0,0]), perturbator=self.perturbator, render=False, reward_shape=self.reward_shaping, normalize=self.normaliz


    def eval(self, genome):

        port = get_free_port()
        env = smartDartEnv(VITE_USim(x_init=[0,0]), perturbator=self.perturbator, render=False, reward_shape=self.reward_shaping, normalize=self.normalize, port=port)
        # env = env_queue.get(block=True)

        # print("environement created ", env)
        g = GRN(genome, self.nin, self.nout)
        g.setup()
        g.warmup(25)
        

        # print("running the environment")
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
        
        # print("fit = ", fit)
        env.close()
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

    # global env_queue
    # env_queue = Manager().Queue()
    # for _ in range(2):
    #     env = smartDartEnv(VITE_USim(x_init=[0,0]), perturbator=perturbator, render=False, reward_shape=reward_shaping, normalize=normalize)
    #     env_queue.put(env)
        
    p = grnSmartDart(2, 0, perturbator, reward_shaping, normalize)

    e = EATMuPlusLambda(nin = p.nin, nout = p.nout, nreg=p.nreg)

    alg, hist = e.run(5, p.eval, 5, 10, multiproc=True, verbose=True)
    e.visualize_evolutions()


    print("best genome is ", alg[0])
    test_environment = smartDartEnv(VITE_USim([0, 0]), perturbator = perturbator, render = args.render, n_stack=args.nstack, normalize = args.normalize, reward_shape=False)
    
    best = GRN(alg[0], p.nin, p.nout)
    fits = []
    for i in range(1):
        
        best.setup()
        best.warmup(25)
        obs, _ = test_environment.reset()
        done = False
        fit = 0
        while not done:
            best.set_input(obs)
            best.step(10)
            action = best.get_output() 
            action = action*80 - 40
            obs, reward, done, truncated, terminated, = test_environment.step(action)

            fit += reward 
            if truncated or terminated: 
                done = True

        fits.append(fit)

    for f in fits:
        print("tested fitness is : ", f)

    pp = test_environment.player_positions
    test_environment.close()
    plt.plot(pp[:, 0], pp[:,1])
    plt.show()

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
