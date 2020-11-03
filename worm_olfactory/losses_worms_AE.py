import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import pdb

class ELBO_loss(torch.nn.Module):
    """
    ELBO loss:
    L = -E_{z~q({X}|{F},{S})}[log(P({F}|{X},{S}))+log(P({X}|{S}))] + KL(q({X}|{F},{S})||Pn)
    loss = reconstruction loss + trajectory loss + KLD
    
    T: time_window for training
    R: interpolation rate: acquisition time/ simulation time
    
    X: X(0), X(1),... X(RT-1)
    F: F(0), F(1),... F(RT-1)
    S: S(0), S(1),... S(RT-1)
    
    ###Reconstruction loss:
    -E_{z~q({X}|{F},{S})}[log(P({F}|{X},{S}))]
    Variables:
    mu_fluorescence_trace_prob
    logvar_fluorescence_trace_prob
    Fluorescence_trace_target
    
    ###Trajectory Loss:
    -E_{z~q({X}|{F},{S})}[log(P({X}|{S}))]
    Variables:
    Outputs from inference network:
    mu of X: mu_neuron_voltage_latent
    logvar of X: logvar_neuron_voltage_latent
    X (sample): sample_neuron_voltage_latent
    mu_neuron_voltage_prob
    logvar_neuron_voltage_prob
    
    ###KLD:
    KL(q({X}|{F},{S})||Pn)
    distance between latent variables of X and N(0,1)
    Variables:
    Outputs from inference network:
    X (sample): sample_neuron_voltage_latent
    mu of X: mu_neuron_voltage_latent
    logvar of X: logvar_neuron_voltage_latent
    """
    

    def __init__(self, N, window_size, R):
    
        #N: Total neuron number in connectome
        #window_size: T for time_window for training
        #R: interpolation rate: acquisition time/ simulation time

        super(ELBO_loss, self).__init__()

        # To be appended at each training iteration
        self.loss_history_ = [] 
        self.recon_loss_history_ = []
        #self.trajectory_loss_history_ = []
        self.KLD_history_ = []

        # parameter during training
        self.mu_neuron_voltage_init = Parameter(torch.rand(N,1))
        #self.logvar_neuron_voltage_init = Parameter(torch.rand(N,1))
        self.logvar_neuron_voltage_init_0 = Parameter(torch.rand(N,1))
        
        self.logvar_fluorescence_trace_prob = Parameter(torch.rand(N,window_size * R))
        #self.logvar_neuron_voltage_prob = Parameter(torch.rand(N,window_size * R))
        #self.logvar_neuron_voltage_prob_0 = Parameter(0.01 * torch.rand(N,window_size * R))
        self.logvar_neuron_voltage_prob_0 = Parameter(0 * torch.rand(N,window_size * R))
        
    def forward(self, fluorescence_trace_target, missing_fluorescence_target, downsample_factor, mu_neuron_voltage_prob, mu_fluorescence_trace_prob, mu_neuron_voltage_latent, logvar_neuron_voltage_latent, sample_neuron_voltage_latent, recon_weight = 1, KLD_weight = 1, variance_weight =1):
        # variables' sizes (N, TR)
        
        # reconstruction loss: log_likelihood
        # log(P({F}|{X},{S})), scalar average on (N, T * R)
        # variance manually decay
        #std_neuron_voltage_prob = 0.1 * F.sigmoid(self.logvar_neuron_voltage_prob_0)
        #std_neuron_voltage_init = 0.1 * F.sigmoid(self.logvar_neuron_voltage_init_0)
        std_neuron_voltage_prob = 0 * F.softplus(self.logvar_neuron_voltage_prob_0)
        std_neuron_voltage_init = F.softplus(self.logvar_neuron_voltage_init_0)
        #self.logvar_neuron_voltage_prob = torch.log(std_neuron_voltage_prob**2)
        self.logvar_neuron_voltage_prob = torch.zeros_like(std_neuron_voltage_prob)
        #self.logvar_neuron_voltage_init = torch.log(std_neuron_voltage_init**2)
        self.logvar_neuron_voltage_init = torch.zeros_like(std_neuron_voltage_init)
        
        # prior: normal distribution, without connectomics constraints
        #self.logvar_neuron_voltage_init = torch.zeros_like(self.logvar_neuron_voltage_init_0)
        #self.mu_neuron_voltage_init = torch.zeros_like(self.mu_neuron_voltage_init_0)
        #self.logvar_neuron_voltage_prob = torch.zeros_like(self.logvar_neuron_voltage_prob_0)
        #mu_neuron_voltage_prob = torch.zeros_like(mu_neuron_voltage_prob)

        std_fluorescence_trace_prob = torch.exp(0.5*self.logvar_fluorescence_trace_prob)
        std_neuron_voltage_prob = torch.exp(0.5*self.logvar_neuron_voltage_prob)
        missing_ratio = torch.sum(missing_fluorescence_target)/(missing_fluorescence_target.shape[0]*missing_fluorescence_target.shape[1]*missing_fluorescence_target.shape[2])
        
        recon_log_likelihood = -torch.mean((torch.distributions.normal.Normal(mu_fluorescence_trace_prob[:,:,::downsample_factor],std_fluorescence_trace_prob[None,:,::downsample_factor]).log_prob(fluorescence_trace_target))* (1 - missing_fluorescence_target))/(1 - missing_ratio)
        
        # trajectory probability distrbution: log_likelihood
        # log(P({X}|{S})), scalar average on (N, T * R)
        # std_neuron_voltage_init = torch.exp(0.5 * self.logvar_neuron_voltage_init)
        # trajectory_log_likelihood = -torch.mean((torch.sum(torch.distributions.normal.Normal(mu_neuron_voltage_prob[:,:,1:],std_neuron_voltage_prob[:,:,1:]).log_prob(sample_neuron_voltage_latent[:,:,1:]),1) + torch.distributions.normal.Normal(self.mu_neuron_voltage_init[None,:],std_neuron_voltage_init[None,:]).log_prob(sample_neuron_voltage_latent[:,:,0]))/mu_neuron_voltage_prob.shape[2])
        
        # prior: normal distribution, without connectomics constraints
        #self.logvar_neuron_voltage_init = torch.zeros_like(self.logvar_neuron_voltage_init_0)
        #self.mu_neuron_voltage_init = torch.zeros_like(self.mu_neuron_voltage_init)
        #self.logvar_neuron_voltage_prob = torch.zeros_like(self.logvar_neuron_voltage_prob_0)
        #mu_neuron_voltage_prob = torch.zeros_like(mu_neuron_voltage_prob)
        
        # KL divergence: KL(q({X}|{F},{S})||Pn), Pn is a learned prior instead of N(0,1)
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # closed form:
        KLD_latent = 0.5 * torch.mean((torch.sum(self.logvar_neuron_voltage_prob[None,:,1:] - logvar_neuron_voltage_latent[:,:,1:] + (logvar_neuron_voltage_latent[:,:,1:].exp() + ((mu_neuron_voltage_latent[:,:,1:] - mu_neuron_voltage_prob[:,:,1:]).pow(2)))/(self.logvar_neuron_voltage_prob[None,:,1:].exp()) - 1,2) + self.logvar_neuron_voltage_init[None,:] - logvar_neuron_voltage_latent[:,:,0] + (logvar_neuron_voltage_latent[:,:,0].exp() + ((mu_neuron_voltage_latent[:,:,0] - self.mu_neuron_voltage_init[None,:]).pow(2)))/(self.logvar_neuron_voltage_init[None,:].exp()) - 1)/mu_neuron_voltage_prob.shape[2])
        
        loss = recon_weight * recon_log_likelihood + KLD_weight * KLD_latent
        
        self.loss_history_.append(loss.item())
        self.recon_loss_history_.append(recon_log_likelihood.item())
        #self.trajectory_loss_history_.append(trajectory_log_likelihood.item())
        self.KLD_history_.append(KLD_latent.item())
        
        return loss, recon_log_likelihood, KLD_latent  #scalar
        


