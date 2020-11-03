import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import numpy as np


class LeakyIntegrator(Module):
    def __init__(self, connectome, dt, hidden_init=None, hidden_init_trainable=False,  bias_init=None, tau_init=None, nonlinearity=F.relu):
        super(LeakyIntegrator, self).__init__()
        self.dt = dt
        self.n = connectome.n
        self.nonlinearity = nonlinearity
        self.sparsity_c = Parameter(torch.Tensor(connectome.sparsity_c), requires_grad=False)
        self.sparsity_e = Parameter(torch.Tensor(connectome.sparsity_e), requires_grad=False)
        self.signs_c = Parameter(torch.Tensor(connectome.signs_c))
        self.magnitude_scaling_factor = 1e-6

        self.magnitudes_c = Parameter(self.magnitude_scaling_factor*torch.Tensor(connectome.magnitudes_c))
        self.magnitudes_e = Parameter(self.magnitude_scaling_factor*torch.Tensor(connectome.magnitudes_e))

        if bias_init is None:
            self.bias = Parameter(torch.empty(self.n).uniform_(0.01, 0.02))
        else:
            self.bias = Parameter(bias_init)
        if tau_init is None:
            self.tau = Parameter(torch.empty(self.n).uniform_(self.dt, 0.2))
        else:
            self.tau = Parameter(tau_init)

        if hidden_init is None:
            hidden_init = torch.zeros(self.n)
        self.hidden_init = Parameter(torch.Tensor(hidden_init), requires_grad=hidden_init_trainable)


    def forward(self, input, custom_init=None):
        with torch.no_grad():
            # time scales, initializations, and synaptic weights must all be positive
            self.magnitudes_c.data.clamp_(min = 0)
            self.magnitudes_e.data.clamp_(min = 0)
            #self.tau.data.clamp_(min = self.dt)

        W_c = torch.mul(self.sparsity_c, self.magnitudes_c)
        W_c = torch.mul(W_c, self.signs_c)
        W_e = torch.mul(self.sparsity_e, (self.magnitudes_e + self.magnitudes_e.transpose(0,1)))
        tau_clamp = self.tau.clamp(min=self.dt)
        timesteps = input.shape[0]

        recurrent_in = []
        external_in = []    
        if custom_init is not None:
            hidden_states = [custom_init]
        else:
            hidden_states = [self.hidden_init]
        for t in range(timesteps):
            x = hidden_states[-1]
            chem_in = torch.mm(W_c, self.nonlinearity(x).transpose(0,1)).transpose(0,1)
            gap_potentials = x - x.t()
            elec_in = torch.sum(torch.mul(W_e, gap_potentials),dim = 1)
            x_tp1 = (self.dt / tau_clamp) * (chem_in + elec_in + self.bias + input[t, :] - x) + x

            hidden_states.append(x_tp1)
            recurrent_in.append(chem_in + elec_in)
            external_in.append(input[t,:].unsqueeze(0))
        return torch.cat(hidden_states[1:], dim=0), torch.cat(recurrent_in, dim=0), torch.cat(external_in, dim = 0)

