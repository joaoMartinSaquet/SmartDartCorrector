import numpy as np


class Corrector():
    def __init__(self, learn = False):
        self.learning = learn
        self.feedback_list = []
        self.step = 0

    def reset(self):

        self.step = 0
        self.feedback_list = []


    def __call__(self, input):
        return input
    

    def get_feedback(self, fb):
        """function register the feedback that occurs inside the env (reward, done, etc...)

        Args:
            fb (_type_): _description_
        """

        self.feedback_list = []



class LowPassCorrector(Corrector):
    ''' averages overs 5 steps and then applies the correction'''
    def __init__(self, nsteps = 5,learn = False):
        super().__init__(learn)
        self.feedback_list = []
        self.step = 0
        self.inputs = deque(maxlen = nsteps)
        self.nsteps = nsteps

    def __call__(self, input):
        self.inputs.append(input)

        if len(self.inputs) == self.nsteps:
            return np.mean(self.inputs, axis=0)
        else:
            return input    
        
        