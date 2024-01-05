import torch
import torch.nn as nn
from torch.autograd import Variable


class Architect(object):
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model
        self.optimizer = optimizer
    
    def log_learning_rate(self, logger):
        if self.args.log:
            logger.info("Architecture Learning Rate: {}".format(lr))
               
    def step(self, inputs):
        loss, outputs = self.model(inputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        lr = self.optimizer.param_groups[0]['lr']
        return loss, lr
