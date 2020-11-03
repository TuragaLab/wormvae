import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import numpy as np
import pdb

class leaky_integrator_VAE_worms_deterministic(Module):
    def __init__(self, connectome, dt, hidden_init=None, hidden_init_trainable=False,  bias_init=None, tau_init=None, nonlinearity=F.relu, is_Training = True):
        super(leaky_integrator_VAE_worms_deterministic, self).__init__()
        self.dt = dt
        self.n = connectome.n
        self.nonlinearity = nonlinearity
        '''
        # connectome matrix shuffle
        magnitudes_c = connectome.magnitudes_c.view(-1)
        np.random.seed(1)
        np.random.shuffle(magnitudes_c)
        magnitudes_c=magnitudes_c.reshape(self.n,self.n)
        magnitudes_e = connectome.magnitudes_e.view(-1)
        np.random.seed(1)
        np.random.shuffle(magnitudes_e)
        magnitudes_e=magnitudes_e.reshape(self.n,self.n)
        
        sparsity_c = connectome.sparsity_c.view(-1)
        np.random.seed(1)
        np.random.shuffle(sparsity_c)
        sparsity_c=sparsity_c.reshape(self.n,self.n)
        sparsity_e = connectome.sparsity_e.view(-1)
        np.random.seed(1)
        np.random.shuffle(sparsity_e)
        sparsity_e=sparsity_e.reshape(self.n,self.n)
        '''
        #self.sparsity_c = Parameter(torch.Tensor(sparsity_c), requires_grad=False)
        #self.sparsity_e = Parameter(torch.Tensor(sparsity_e), requires_grad=False)
        self.sparsity_c = Parameter(torch.Tensor(connectome.sparsity_c), requires_grad=False)
        self.sparsity_e = Parameter(torch.Tensor(connectome.sparsity_e), requires_grad=False)
        self.signs_c = Parameter(torch.Tensor(connectome.signs_c))
        #self.magnitude_scaling_factor = 1e-4
        self.magnitude_scaling_factor = 1e-2
        #self.magnitude_scaling_factor = 1e-1
        #self.magnitude_scaling_factor = 1
        self.is_Training = is_Training
        
        #self.magnitudes_c = Parameter(self.magnitude_scaling_factor*torch.Tensor(magnitudes_c))
        #self.magnitudes_e = Parameter(self.magnitude_scaling_factor*torch.Tensor(magnitudes_e))
        
        self.magnitudes_c = Parameter(self.magnitude_scaling_factor*torch.Tensor(connectome.magnitudes_c))
        self.magnitudes_e = Parameter(self.magnitude_scaling_factor*torch.Tensor(connectome.magnitudes_e))
        
        self.prelu_layer = torch.nn.PReLU()
        
        if bias_init is None:
            self.bias = Parameter(torch.empty(self.n).uniform_(0.01, 0.02))
            #self.bias = Parameter(torch.empty(self.n).uniform_(1, 2))
        else:
            self.bias = Parameter(bias_init)
        if tau_init is None:
            self.tau = Parameter(torch.empty(self.n).uniform_(self.dt, 0.2))
        else:
            self.tau = Parameter(tau_init)

        if hidden_init is None:
            hidden_init = torch.zeros(self.n)
        self.hidden_init = Parameter(torch.Tensor(hidden_init), requires_grad=hidden_init_trainable)
        
    def forward(self, input, hidden_infer_states, T_steps = 400):
        with torch.no_grad():
            # time scales, initializations, and synaptic weights must all be positive
            self.magnitudes_c.data.clamp_(min = 0)
            self.magnitudes_e.data.clamp_(min = 0)
            #self.tau.data.clamp_(min = self.dt)

        W_c = torch.mul(self.sparsity_c, self.magnitudes_c)
        W_c = torch.mul(W_c, self.signs_c)
        W_e = torch.mul(self.sparsity_e, (self.magnitudes_e + self.magnitudes_e.transpose(0,1)))
        tau_clamp = self.tau.clamp(min=self.dt)
        timesteps = input.shape[2]

        recurrent_in = []
        external_in = []
        hidden_states = []
        for t in range(timesteps):
            if t ==0:
                x = hidden_infer_states[0,:,t].unsqueeze(0)
                chem_in = torch.zeros(1,input.shape[1])
                elec_in = torch.zeros(1,input.shape[1])
                hidden_states.append(x)
            elif t%T_steps ==0:
                x = hidden_states[-1]
                chem_in = torch.mm(W_c, self.nonlinearity(x).transpose(0,1)).transpose(0,1)
                gap_potentials = x - x.t()
                elec_in = torch.sum(torch.mul(W_e, gap_potentials),dim = 1)
                # for every T_step, take inferred voltage
                x_tp1 = hidden_infer_states[0,:,t].unsqueeze(0)
                hidden_states.append(x_tp1)
            else:
                x = hidden_states[-1]
                chem_in = torch.mm(W_c, self.nonlinearity(x).transpose(0,1)).transpose(0,1)
                gap_potentials = x - x.t()
                elec_in = torch.sum(torch.mul(W_e, gap_potentials),dim = 1)
                x_tp1 = (self.dt / self.tau) * (chem_in + elec_in + self.bias + input[0,:,t] - x) + x
                hidden_states.append(x_tp1)
            recurrent_in.append(chem_in + elec_in)
            external_in.append(input[0,:,t].unsqueeze(0))
        return torch.cat(hidden_states, dim=0).T.unsqueeze(0), torch.cat(recurrent_in, dim=0).T.unsqueeze(0)

    def stochatics_forward(self, input, hidden_states):
        with torch.no_grad():
            # time scales, initializations, and synaptic weights must all be positive
            self.magnitudes_c.data.clamp_(min = 0)
            self.magnitudes_e.data.clamp_(min = 0)
            #self.tau.data.clamp_(min = self.dt)

        W_c = torch.mul(self.sparsity_c, self.magnitudes_c)
        W_c = torch.mul(W_c, self.signs_c)
        W_e = torch.mul(self.sparsity_e, (self.magnitudes_e + self.magnitudes_e.transpose(0,1)))
        #tau_clamp = self.tau.clamp(min=self.dt)
        timesteps = input.shape[2]

        x = torch.zeros(hidden_states.shape)
        # x: initial states, (batch, n, R * T)
        x[:,:,1:] = hidden_states[:,:,:-1]
        batch_W_c = W_c.unsqueeze(0).repeat(x.shape[0],1,1)
        chem_in = torch.bmm(batch_W_c, self.nonlinearity(x)) #chem_in: (batch, n, R * T)
        #using prelu with learnable weight params
        #chem_in = torch.bmm(batch_W_c, self.prelu_layer(x)) #chem_in: (batch, n, R * T)
        x_trans = x.permute(0,2,1)[:, :, None, :] #(batch, R * T, 1 ,n)
        if self.is_Training:
            gap_potentials = x_trans - x_trans.permute(0,1,3,2) #(batch, R * T, n, n)
            elec_in = torch.sum(torch.mul(W_e[None,None,:,:], gap_potentials), dim = 3).permute(0,2,1) #elec_in: (batch, n, R * T)
        else:
            elec_in = torch.zeros(input.shape)
            # predict: batch_size: 1
            for it in range(timesteps):
                x_t = x[:,:,it][:,None,:] # (batch, 1, n)
                gap_potentials = x_t - x_t.permute(0,2,1) #(batch, n, n)
                elec_in[:,:,it] = torch.sum(torch.mul(W_e[None,:,:], gap_potentials), dim = 2) #elec_in: (batch, n, R * T)
        # input (sensory) (batch, n, R* T)
        dt_tau = (self.dt / self.tau).squeeze(0)
        bias = self.bias.squeeze(0)
        recurrent_in = chem_in + elec_in #(batch, n, R * T)
        mu_neuron_voltage_prob = torch.mul(dt_tau[None,:, None], recurrent_in + bias[None, :, None] + input - x) + x #(batch, n, R * T)
        #(batch, n, R * T)
        return mu_neuron_voltage_prob, recurrent_in

