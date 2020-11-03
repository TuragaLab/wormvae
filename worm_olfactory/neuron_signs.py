import torch.nn as nn
import torch

class Network_Signs(nn.Module):
    def __init__(self, N_neurons, random_seed = None):
        super(Network_Signs, self).__init__()
        torch.random.manual_seed(random_seed)
        self.neuron_signs = torch.nn.Parameter(torch.rand((1, N_neurons), dtype = torch.float32, device = 'cuda')-0.2)