import torch.nn as nn
import torch

class Synapse_Signs(nn.Module):
    def __init__(self, N_neurons, random_seed = None):
        super(Synapse_Signs, self).__init__()
        torch.random.manual_seed(random_seed)
        self.synapse_signs = torch.nn.Parameter(torch.rand((N_neurons, N_neurons), dtype = torch.float32, device = 'cuda')-0.2)