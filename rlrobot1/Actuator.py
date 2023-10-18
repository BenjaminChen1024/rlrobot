# Actuator Network - Menzi Muck M545
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from rlrobot.algorithms.dl.mlp.mlp import MLPLayer

class ActuatorNet():
    """
    Actuator Network    class.  
    The actuator model predicts the joint velocities at the next time step, 
    given the state at the current time step and a history of past states.

    :param jt_pos: Joint position (current position)
    :param jt_v: Joint velocities (current velocities, a history of past joint velocities)
    :param valve_setpoints: Valve setpoints (a history of the input commands over the past 0.99s)
    :param rpm: Diesel engine rpm
    :param temp: Hydraulic oil temperature
    """

    def __init__(self, jt_pos, jt_v, valve_setpoints, rpm, temp):
        
        self.jt_pos = jt_pos
        self.jt_v = jt_v
        self.valve_setpoints = valve_setpoints
        self.rpm = rpm
        self.temp = temp

        self.batch_size = 64
        self.model = MLPLayer(5, 1, 128, 3)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.adam()

    def train(self):
        self.model.train()
        for 


if __name__ == '__main__':
    actuator
    Actuator = ActuatorNet()

