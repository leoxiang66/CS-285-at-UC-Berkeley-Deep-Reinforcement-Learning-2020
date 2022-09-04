import torch.nn as nn

class BaseReward(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def compute_reward(self, state, action):
        raise NotImplementedError
