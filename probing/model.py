import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F

class ProbeLayer(Module):
    """
     Linear probe for classification on layer to see if it contains relevanti information on the task
    """
    
    def __init__(self, input_dim, num_classes):
        super(ProbeLayer, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.linear(x)
        return x