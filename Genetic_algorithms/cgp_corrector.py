# cgp imports
from pyCGP.pycgp import cgp, evaluators, viz, cgpfunctions
from pyCGP.pycgp.cgp import *
from pyCGP.pycgp.cgpfunctions import *
from pyCGP.pycgp.cgpes import *

# corrector imports
from common.user_simulator import *
from common.perturbation import *
from common.rolloutenv import *
from common.corrector import *

# usual imports
import numpy as np
import sympy as sp


FUN_LIB =  [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            # CGPFunc(f_max, 'div', 2, 0, '/'),
            CGPFunc(f_gt, 'gt', 2, 0, '>'),
            CGPFunc(f_log, 'log', 1, 0, 'log'),
            CGPFunc(f_sqrtxy, 'sqrtxy', 2, 0, 'sqt'),
            CGPFunc(f_const, 'c', 0, 1, 'c')
            ]


class CGP_corrector(evaluators.Evaluator, Corrector):
    def __init__(self, nstep, col, row, perturbator = None):
        super().__init__()

        self.fun_lib = FUN_LIB

        self.input_shape = (1, 2) # dx and dy 
        self.output_shape = (1, 2)# dx and dy estimated

        self.n_input = 2
        self.n_output = 2
    
        self.col = col
        self.row = row

    
        
        self.perturbator = perturbator
        self.nstep = nstep

        self.env = GodotEnv(convert_action_space=True)

    def evaluate(self, cgp, it):
        
        individual_corr = cgp.run()
        rgathered, _= rolloutSmartDartEnv(env, self.nstep, self.perturbator, corrector = individual_corr)
        return rgathered

    def evolve(self, mu, nb_ind = 4, num_csts = 0, 
               mutation_rate_nodes = 0.2, mutation_rate_outputs=0.2, mutation_rate_const_params=0.01,
               n_cpus=1, n_it=20, folder_name='test', term_criteria=20, random_genomes=True):
        
        
        self.hof = [CGP_with_cste.random(self.n_inputs, self.n_outputs, num_csts, self.col, self.row, self.library, 
                                    self.col, False, const_min=-10, const_max=10, input_shape=self.input_shape, dtype='float')
                                        for i in range(mu)]

        # not used yet                
        # if not random_genomes:
        #     # get the prior knowledge of the problem (fitts dx = y0 and dy = y1)
        #     for ind in self.hof:
        #         ind.genome[-2] = 0 # y0 = dx
        #         ind.genome[-1] = 1 # y1 = dy
        es = CGPES_ml(mu, nb_ind, mutation_rate_nodes, mutation_rate_outputs, mutation_rate_const_params, self.hof, self, folder_name, n_cpus)
        es.run(n_it, term_criteria=term_criteria)
        
        fit_history = es.fitness_history
        best = es.hof[np.argmax(es.hof_fit)]
        
        self.best = best
        return best, fit_history
    
    def clone(self):
        return CGP_corrector(self.nstep, self.col, self.row, self.perturbator)
    
    def best_logs(self, input_names, output_names):
        
        
        out, infix_out = self.best.to_function_string(input_names, output_names)
    
        print("best raw equations : ",out)
        out_equation = []
        for o in infix_out:
            print("HOF best simplified : ", sp.simplify(o))
            out_equation.append(sp.simplify)

        return out_equation

if __name__ == "__main__":

    print("testing pycgp")