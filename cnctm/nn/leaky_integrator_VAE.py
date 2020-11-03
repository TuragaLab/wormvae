import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import numpy as np
import pdb


class leaky_integrator_VAE(Module):
    def __init__(self, connectome, dt, hidden_init=None, hidden_init_trainable=False,  bias_init=None, tau_init=None, nonlinearity=F.relu, is_Training = True):
        super(leaky_integrator_VAE, self).__init__()
        self.dt = dt
        self.n = connectome.n
        self.nonlinearity = nonlinearity
        self.sparsity_c = Parameter(torch.Tensor(connectome.sparsity_c), requires_grad=False)
        self.sparsity_e = Parameter(torch.Tensor(connectome.sparsity_e), requires_grad=False)
        self.signs_c = Parameter(torch.Tensor(connectome.signs_c))
        self.magnitude_scaling_factor = 1e-6
        self.is_Training = is_Training

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


    def forward(self, input, hidden_states):
        with torch.no_grad():
            # time scales, initializations, and synaptic weights must all be positive
            self.magnitudes_c.data.clamp_(min = 0)
            self.magnitudes_e.data.clamp_(min = 0)
            #self.tau.data.clamp_(min = self.dt)

        W_c = torch.mul(self.sparsity_c, self.magnitudes_c)
        W_c = torch.mul(W_c, self.signs_c)
        W_e = torch.mul(self.sparsity_e, (self.magnitudes_e + self.magnitudes_e.transpose(0,1)))
        #tau_clamp = self.tau.clamp(min=self.dt)
        timesteps = input.shape[1]

        x = torch.zeros(input.shape)
        # x: initial states, (n, R* window_size)
        x[:,1:] = hidden_states[:,:-1]
        chem_in = torch.mm(W_c, self.nonlinearity(x)) #chem_in: (n, R* window_size)
        x_trans = x.transpose(0,1)[:,None,:] #(R* window_size,1,n)
        gap_potentials = x_trans - x_trans.transpose(1,2) #(R* window_size,n,n)
        if self.is_Training:
            elec_in = torch.sum(torch.mul(W_e[None,:,:], gap_potentials), dim = 2).transpose(0,1) #elec_in: (n, R* window_size)
        else:
            elec_in = torch.zeros(input.shape)
            for it in range(timesteps):
                x_t = x[:,it].unsqueeze(0) #(1,n)
                gap_potentials = x_t - x_t.t() #(n,n)
                elec_in[:,it] = torch.sum(torch.mul(W_e, gap_potentials), dim = 1) #elec_in: (n, R* window_size)
        # input (sensory) (n, R* window_size)
        dt_tau = (self.dt / self.tau).squeeze(0)
        bias = self.bias.squeeze(0)
        recurrent_in = chem_in + elec_in #(n, R* window_size)
        mu_neuron_voltage_prob = torch.mul(dt_tau[:,None], recurrent_in + bias[:,None] + input - x) + x #(n, R* window_size)
        
        return mu_neuron_voltage_prob, recurrent_in

