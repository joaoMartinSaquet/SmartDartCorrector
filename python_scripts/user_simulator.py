import numpy as np

WIDTH_TARGET = 20


class UserSimulator():
    def __init__(self):
        pass

    def step(self):
        pass

    def reset(self):
        pass


class VITE_USim(UserSimulator):
    def __init__(self, x_init, dt = 0.008):
        """the controller acting as a closed loops, we needs to compute the error in 2D and then return the displacement actions

            this controller is control as xd = Ax + Bu
                                          y = Cx + Du

            X is [x, y, xdot, ydot]
            xd is [xdd, ydd]
            Y is xdisp and ydisp
            xd = 
        """
        # Non working dynamics but could be interesiting to see if we can make it works with a controller behind 
        # self.gamma = 0.08
        # # x position system [x, xdot]

        # # self.Ax = np.array([[0, 1], [-1, -1]]) * self.gamma OLD
        # self.Ax = np.array([[0, 1], [-self.gamma, -self.gamma]]) 
        # # self.Bx = np.array([[1], [0]])/12.5 OLD
        # self.Bx = np.array([[0], [1]])*self.gamma


        # # self.Ay = np.array([[0, 1], [-1, -1]]) * self.gamma OLD
        # self.Ay = np.array([[0, 1], [-self.gamma, -self.gamma]]) 
        # # self.By = np.array([[1], [0]])/12.5
        # self.By = np.array([[0], [1]]) * self.gamma





        self.gamma = 0.08
        # x position system [x, xdot]

        self.Ax = np.array([[0, 1], [-1, -1]]) * self.gamma
        self.Bx = np.array([[1], [0]])/12.5


        self.Ay = np.array([[0, 1], [-1, -1]]) * self.gamma 
        self.By = np.array([[1], [0]])/12.5

        self.xx = [x_init[0], 0]
        self.xy = [x_init[1] , 0]

        self.dt = 1


    def reset(self, x_init):
        
        self.xx = [x_init[0], 0]
        self.xy = [x_init[1] , 0]
        
        

    def step(self, y_target, y_process):
        """_summary_

        Args:
            y_target (_type_): _description_
            y_process (_type_): should be equals to x[1,2]

        Returns:
           
            
        """

        err = y_target - y_process # is the input
        if np.linalg.norm(err) < WIDTH_TARGET:
            click_action = 1
        else :
            click_action = 0
        

        # x system
        # recontruct x with the x observed
        self.xx[0] = y_process[0] 
        xxdot = self.Ax @ self.xx + self.Bx @ [y_target[0]]
        self.xx = self.xx + xxdot*self.dt

        # y system
        # recontruct x with the x observed
        self.xy[0] = y_process[1] 
        xydot = self.Ay @ self.xy + self.By @  [y_target[1]]
        self.xy= self.xy + xydot*self.dt


        action = [xxdot[0] , xydot[0]]

        return action, click_action