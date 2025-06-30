# cgp imports
from GA.pyCGP.pycgp import cgp, evaluators, viz, cgpfunctions
from GA.pyCGP.pycgp.cgp import *
from GA.pyCGP.pycgp.cgpfunctions import *
from GA.pyCGP.pycgp.cgpes import *

# corrector imports
from common.user_simulator import *
from common.perturbation import *
from common.rolloutenv import *
from common.corrector import *

# usual imports
import numpy as np
import sympy as sp
import time
import matplotlib.pyplot as plt
from collections import deque
from loguru import logger
import pandas as pd

LOG_PATH = "logs_corrector/CGP/" + time.strftime("%Y%m%d-%H%M%S") 

if not os.path.exists(LOG_PATH):
    logger.info("creating log folder at : ", LOG_PATH)
    os.makedirs(LOG_PATH)

FUN_LIB =  [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGPFunc(f_max, 'div', 2, 0, '/'),
            # CGPFunc(f_gt, 'gt', 2, 0, '>'),
            CGPFunc(f_log, 'log', 1, 0, 'log'),
            CGPFunc(f_sqrtxy, 'sqrtxy', 2, 0, 'sqt'),
            CGPFunc(f_const, 'c', 0, 1, 'c')
            ]


class CGPEvaluator(evaluators.Evaluator):
    def __init__(self, ngen, nstep, col, row, perturbator = None):
        super().__init__()

        self.fun_lib = FUN_LIB

        self.input_shape = (1, ) # dx and dy 
        self.output_shape = (1, )# dx and dy estimated

        self.n_inputs = 2
        self.n_outputs = 2
    
        self.col = col
        self.row = row

        self.ngen = ngen 
        
        self.perturbator = perturbator
        self.nstep = nstep



    def evaluate(self, cgp, it):

        individual_corr = cgp.run
        env = envs.pop()
        rgathered, _= rolloutSmartDartEnv(env, self.nstep, self.perturbator, corrector = individual_corr, log = 0)
        envs.append(env)
        return rgathered

    
    def clone(self):
        return CGPEvaluator(self.ngen, self.nstep, self.col, self.row, self.perturbator)

class CGPCorrector(Corrector):
    def __init__(self, env, ngen, nstep, col, row, perturbator = None):
        super().__init__()

        self.fun_lib = FUN_LIB

        self.input_shape = (1, ) # dx and dy 
        self.output_shape = (1, )# dx and dy estimated

        self.n_inputs = 2
        self.n_outputs = 2
    
        self.col = col
        self.row = row

        self.ngen = ngen 
        
        self.perturbator = perturbator
        self.nstep = nstep
        
        self.env = env

        if isinstance(self.env, StableBaselinesGodotEnv) and self.env.num_envs > 1:
            self.sb = True
            # create the list of envs
            global envs
            envs = deque(self.env.envs, maxlen=self.env.num_envs)   
        else:
            self.sb = False

        self.evaluator = CGPEvaluator(self.ngen, self.nstep, self.col, self.row, self.perturbator)

    
    
    def best_logs(self, input_names, output_names):
        
        
        out, infix_out = self.best.to_function_string(input_names, output_names)
    
        logger.debug("best raw equations : ",out)
        out_equation = []
        for o in infix_out:
            # logger.debug("HOF best simplified : ", sp.simplify(o))
            out_equation.append(sp.simplify)

        return out_equation
    
    def learn(self, mu = 4, nb_ind = 4, num_csts = 1, 
               mutation_rate_nodes = 0.2, mutation_rate_outputs=0.2, mutation_rate_const_params=0.01,
               n_cpus=6, n_it=500, folder_name=LOG_PATH, term_criteria=-np.inf, random_genomes=False):
        
        # self.evaluator.evolve(1, num_csts=1, n_it=self.ngen, folder_name=LOG_PATH, random_genomes=False, n_cpus=self.env.num_envs)
        self.hof = [CGP_with_cste.random(self.n_inputs, self.n_outputs, num_csts, self.col, self.row, self.fun_lib, 
                                    self.col, False, const_min=-10, const_max=10, input_shape=self.input_shape, dtype='float')
                                        for i in range(mu)]        
       
        # not used yet                
        if not random_genomes:
            logger.debug("getting prior knowledge of the problem")
            # get the prior knowledge of the problem (fitts dx = y0 and dy = y1)
            for ind in self.hof:
                ind.genome[-2] = 0 # y0 = dx
                ind.genome[-1] = 1 # y1 = dy


        logger.info("evolving m + l" )
        logger.info("hyperparameters : mu {mu}, lamba : {nb_ind}, ngen {ngen}, nstep {nstep}, col {col}, row {row}"
                    .format(mu=mu, nb_ind=nb_ind, ngen=self.ngen, nstep=self.nstep, col=self.col, row=self.row))
        es = CGPES_ml(mu, nb_ind, mutation_rate_nodes, mutation_rate_outputs, mutation_rate_const_params, 
                      self.hof, self.evaluator, folder_name, n_cpus)
                
        es.run(self.ngen, term_criteria=term_criteria)
        
        fit_history = es.fitness_history
        best_idx = np.argmax(es.hof_fit)
        self.best = es.hof[best_idx]
        self.best_genomes = es.best_genomes
        
        input_names = ['dx', 'dy']
        output_names = ['dx_est', 'dy_est']
        try : 
            out = self.best_logs(input_names, output_names)
        except :
            out = "no equation found"
        
        logger.info("best equations : ", out)
        
        G = self.best.netx_graph(input_names, output_names)
        viz.draw_net(G, self.n_inputs, self.n_outputs)
        plt.savefig(LOG_PATH + "/graph.png")

        # logger.info("fit history : ", fit_history)
        # print("best genome : ", self.best_genomes)
        to_dump = {"Mean fit history" : np.mean(fit_history),
        "           best_fit_history" : fit_history,
                   "best_genome" : self.best_genomes}
        
        pd.DataFrame(to_dump).to_csv(LOG_PATH + "/log.csv")
        
        return to_dump

