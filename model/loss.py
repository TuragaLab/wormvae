import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

class ELBO_loss(torch.nn.Module):
    """Evidence lower bound (ELBO)
    Attributes:
        inputs:
            reconstructed fluorescence
            target fluorescence
            voltage prior distribution
            voltage posterior distribution
        ouputs:
            ELBO (objective function)
            reconstruction loss
            kl divergence
    Methods:
        ELBO loss = reconstruction loss + KLD
        E_v~Q(v|f,o)[log(P(f|v, o))] + KL(Q(v|f,o)||P(v|o))

        Reconstruction loss: 
        data likelihood of measured fluorescence trace from target fluorescence and reconstructed fluorescence distribution
        E_v~Q(v|f,o)[log(P(f|v, o))]
        
        KL divergence:
        distance between voltage prior distribution and voltage posterior distribution
        KL(Q(v|f,o)||P(v|o))
    """
    

    def __init__(self, N, window_size, upsample_factor):
        super(ELBO_loss, self).__init__()
        self.window_size = window_size
        self.upsample_factor = upsample_factor
        self.eps = 1e-8
        self.prior_voltage_mu_init = Parameter(2*torch.rand(N,1)-1)
        self.prior_voltage_logvar_init = Parameter(2*torch.rand(N,1)-1)
        self.pred_fluorescence_logvar = Parameter(torch.rand(N,1)-1)
        self.prior_voltage_logvar = Parameter(torch.rand(N,1)-1)

    def forward(self, fluorescence_trace_target, missing_fluorescence_target, outputs, recon_weight = 1, KLD_weight = 1):
  
        simu_steps = self.window_size * self.upsample_factor
        pred_fluorescence_logvar = self.pred_fluorescence_logvar.repeat(1,simu_steps)
        prior_voltage_logvar = self.prior_voltage_logvar.repeat(1,simu_steps)
        prior_voltage_logvar_init = self.prior_voltage_logvar_init
        pred_fluorescence_std = torch.exp(0.5*pred_fluorescence_logvar)

        B, N, record_steps = missing_fluorescence_target.shape
        missing_ratio = torch.sum(missing_fluorescence_target)/(B*N*record_steps)

        # downsample reconstructed fluorescence distribution
        fluorescence_posterior = torch.distributions.normal.Normal(
            outputs['pred_fluorescence_mu'][:,:,::self.upsample_factor],
            pred_fluorescence_std[None,:,::self.upsample_factor])

        # log likelihood for measured fluorescence target 
        recon_log_likelihood = -torch.mean(fluorescence_posterior.log_prob(fluorescence_trace_target) * (1 - missing_fluorescence_target))/(1 - missing_ratio)
        
        # KL divergence t(0)
        KLD_init = self.kl_divergence(
            prior_logvar = self.prior_voltage_logvar_init,
            prior_mu = self.prior_voltage_mu_init,
            posterior_logvar = outputs['voltage_latent_logvar'][:,:,0],
            posterior_mu = outputs['voltage_latent_mu'][:,:,0])

        # KL divergence t(1)...t(T)
        KLD_nexts = self.kl_divergence(
            prior_logvar = prior_voltage_logvar[None, :, 1:],
            prior_mu = outputs['prior_voltage_mu'][:,:,1:],
            posterior_logvar = outputs['voltage_latent_logvar'][:,:,1:],
            posterior_mu = outputs['voltage_latent_mu'][:,:,1:])

        # KL divergence
        KLD_latent = torch.mean((KLD_init + torch.sum(KLD_nexts, 2))/simu_steps)

        # ELBO
        loss = recon_weight * recon_log_likelihood + KLD_weight * KLD_latent
        
        return loss, recon_log_likelihood, KLD_latent
        
    def kl_divergence(self, prior_logvar, prior_mu, posterior_logvar, posterior_mu):
        # KL divergence: KL(q({X}|{F},{S})||Pn)
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # Closed forms between two gaussian distributions 
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

        KLD = prior_logvar - posterior_logvar
        KLD += (posterior_logvar.exp() + (posterior_mu - prior_mu).pow(2))/(prior_logvar.exp() + self.eps)
        KLD += - 1
        KLD *= 0.5
        return KLD

