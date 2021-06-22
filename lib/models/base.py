from abc import ABC, abstractmethod
import torch.nn as nn

class Backbone(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, acc, ppg):
        pass

    def compute_mac_operation(self):
        pass

    def compute_number_of_neurons(self):
        pass
