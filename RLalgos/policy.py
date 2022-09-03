import torch
import torch.nn as nn

class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def produce_action(self,observation):
        raise NotImplementedError