import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import numpy as np

class leaky_integrator_conductance(Module):
    """Leaky integrator for voltage dynamics using conductance-based synapse model
    Attrbutes:
        inputs:
            connectome
            sensory input
            constraint setting (synapse count, synapse sparsity, unconstrained)
        outputs:
            voltage prior distribution
            chemical synaptic input
            electrial synaptic input
    Methods:
        leaky integrator for voltage dynamics
        network dynamics: tau dv(t)/dt + v(t) = chem_in(t) + elec_in(t) + v_rest + sensory_in(t) + std_v * noise_v(t)
        conductance-based chemical synapse model:
            chem_in_i(t) = \sum (Eji-vi(t)) * Wji_c * softplus(vj(t))
            Eji reversal potential (-\infinity, \infinity)
            Wji_c conductance chemical weight [0, \infinity)
        electrical synapse model:
            elec_in_i(t) = \sum Wji_e * (vj(t)-vi(t))
            Wji_e electrical weight (gap junctions) [0, \infinity)
    """
    def __init__(self, connectome, dt, hidden_init=None, hidden_init_trainable=False,  bias_init=None, tau_init=None, nonlinearity=F.relu, constraint = 'sparsity', is_Training = True):
        super(leaky_integrator_conductance, self).__init__()
        self.dt = dt
        self.n = connectome.n
        self.nonlinearity = nonlinearity
        self.is_Training = is_Training
        self.signs_c = Parameter(torch.Tensor(connectome.signs_c))

        magnitude_scaling_factor_chem = 1e-2
        magnitude_scaling_factor_elec = 1e-2
        
        if constraint == 'weight':
            #connectome synapse count constrained
            self.magnitude_scaling_factor_chem = Parameter(torch.tensor(magnitude_scaling_factor_chem))
            self.magnitude_scaling_factor_elec = Parameter(torch.tensor(magnitude_scaling_factor_elec))
            self.sparsity_c = Parameter(torch.Tensor(connectome.sparsity_c), requires_grad=False)
            self.sparsity_e = Parameter(torch.Tensor(connectome.sparsity_e), requires_grad=False)
            self.reversal_potentials = Parameter(torch.empty(1,self.n,self.n).uniform_(-0.05,0.05))
            self.magnitudes_c = Parameter(torch.Tensor(connectome.magnitudes_c), requires_grad=False)
            self.magnitudes_e = Parameter(torch.Tensor(connectome.magnitudes_e), requires_grad=False)

        if constraint =='sparsity':
            #connectome sparsity constrained, connectome synapse count initialize
            self.magnitude_scaling_factor_chem = Parameter(torch.tensor(magnitude_scaling_factor_chem), requires_grad=False)
            self.magnitude_scaling_factor_elec = Parameter(torch.tensor(magnitude_scaling_factor_elec), requires_grad=False)
            self.sparsity_c = Parameter(torch.Tensor(connectome.sparsity_c), requires_grad=False)
            self.sparsity_e = Parameter(torch.Tensor(connectome.sparsity_e), requires_grad=False)
            self.reversal_potentials = Parameter(torch.empty(1,self.n,self.n).uniform_(-0.05,0.05))
            self.magnitudes_c = Parameter(torch.Tensor(connectome.magnitudes_c))
            self.magnitudes_e = Parameter(torch.Tensor(connectome.magnitudes_e))

        if constraint == 'unconstrained':
            #unconstrained
            self.magnitude_scaling_factor_chem = Parameter(torch.tensor(magnitude_scaling_factor_chem), requires_grad=False)
            self.magnitude_scaling_factor_elec = Parameter(torch.tensor(magnitude_scaling_factor_elec), requires_grad=False)
            self.reversal_potentials = Parameter(torch.empty(1,self.n,self.n).uniform_(-0.05,0.05))
            self.sparsity_c = Parameter(torch.ones_like(connectome.sparsity_c), requires_grad=False)
            self.sparsity_e = Parameter(torch.ones_like(connectome.sparsity_e), requires_grad=False)
            self.magnitudes_c = Parameter(torch.FloatTensor(self.n,self.n).uniform_(0, 2 * torch.mean(connectome.magnitudes_c)))
            self.magnitudes_e = Parameter(torch.FloatTensor(self.n,self.n).uniform_(0, 2 * torch.mean(connectome.magnitudes_e)))
        
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

    def forward(self, input, hidden_states, is_Training):

        W_c = torch.mul(self.sparsity_c, self.magnitudes_c * self.magnitude_scaling_factor_chem)
        W_e = torch.mul(self.sparsity_e, (self.magnitudes_e + self.magnitudes_e.transpose(0,1)) * self.magnitude_scaling_factor_elec)
        timesteps = input.shape[2]

        x = torch.zeros(hidden_states.shape)
        x[:,:,1:] = hidden_states[:,:,:-1]
        x_trans = x.permute(0,2,1)[:, :, None, :]
        if is_Training:
            batch_W_c = W_c.unsqueeze(2).repeat(x.shape[0],1,1, x.shape[2])
            batch_reversals = self.reversal_potentials.unsqueeze(3).repeat(1,1,1,x.shape[2])
            gap_potentials = x_trans - x_trans.permute(0,1,3,2)
            chem_in = torch.sum(batch_W_c * (batch_reversals - x.unsqueeze(2).repeat(1,1,self.n,1)), dim = 2).squeeze(0) * self.nonlinearity(x)
            elec_in = torch.sum(torch.mul(W_e[None,None,:,:], gap_potentials), dim = 3).permute(0,2,1)
        else:
            elec_in = torch.zeros(input.shape)
            chem_in = torch.zeros(input.shape)
            batch_W_c = W_c.repeat(x.shape[0],1,1)
            for it in range(timesteps):
                x_t = x[:,:,it][:,None,:]
                chem_in[:,:,it] =  torch.sum(batch_W_c * (self.reversal_potentials - x_t.repeat(1,self.n,1) ), dim = 2).squeeze(0) * self.nonlinearity(x[:,:,it])
                gap_potentials = x_t - x_t.permute(0,2,1)
                elec_in[:,:,it] = torch.sum(torch.mul(W_e[None,:,:], gap_potentials), dim = 2)
        dt_tau = (self.dt / self.tau).squeeze(0)
        bias = self.bias.squeeze(0)
        recurrent_in = chem_in + elec_in
        mu_neuron_voltage_prob = torch.mul(dt_tau[None,:, None], recurrent_in + bias[None, :, None] + input - x) + x
        return mu_neuron_voltage_prob, chem_in, elec_in

