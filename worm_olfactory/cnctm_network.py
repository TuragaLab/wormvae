import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import cnctm.nn as cnctmnn
from cnctm.utils.data import ConnectomeConstructor
import pdb

class Worm_Sensory_Encoder(nn.Module):
    """sensory encoder for odor inputs
    Attrbutes:
    inputs: odor inputs
    outputs: sensory inputs for sensory neurons
    
    Methods:
    linear layers, activation: ReLU
    """
    # odor_inputs: after interpolation: neuron_sensory_num * window_size
    # sensory_inputs (output:): n * window_size

    def __init__(self, n_input, n_output, nonlinearity = F.relu):
        super(Worm_Sensory_Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, n_input)
        self.nonlinearity = nonlinearity
        self.linear2 = torch.nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.linear2(self.nonlinearity(self.linear1(x.T))).T

class Worm_Inference_Network(nn.Module):
    """ inference network for worm_vae_olfactory
    Attributes:
        structure parameters of encoder;
        inputs: fluorescence trace + sensory inputs
        outputs: latent variables
        R: time interpolation rate
    Methods:
        network: 1dconv , linear
        F: neuron_num_record * window_size/R
        latent variables for all neurons:  mu:neuron_num * window_size, sigma: neuron_num * window_size
        encoder: q({X}|{F},{S})
        {S}: sensory inputs
        {X}: latent variables for all neurons
        {F}: fluorescence trace
    """

    def __init__(self, n, record_n, R, ksize = 3, nonlinearity = F.relu):
        # n: neuron num, window_size: window_size
        # R = acqusition_dt/ simulation_dt
        # request: R//5==0
        super(Worm_Inference_Network, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = record_n, out_channels = 2 * n, kernel_size = ksize, padding = 1)
        self.conv2 = torch.nn.Conv1d(in_channels = 2 * n, out_channels = 2 * n, kernel_size = ksize, padding = 1)
        self.conv3 = torch.nn.Conv1d(in_channels = n, out_channels = 2 * n, kernel_size = ksize, padding = 1)
        self.conv4 = torch.nn.Conv1d(in_channels = 4 * n, out_channels = n, kernel_size = ksize, padding = 1)
        self.conv5 = torch.nn.Conv1d(in_channels = 4 * n, out_channels = n, kernel_size = ksize, padding = 1)
        self.nonlinearity = nonlinearity
        self.upsample1 = torch.nn.Upsample(scale_factor= 5,mode='linear', align_corners=True)
        self.upsample2 = torch.nn.Upsample(scale_factor= int(R/5),mode='linear', align_corners=True)
        
    def forward(self, fluorescence_trace_target, sensory_input):
        # inputs:
        # fluorescence_trace_target (n, window_size)
        # sensory_input (n, window_size)
        # batch_size = 1
        y1 = self.nonlinearity(self.conv1(fluorescence_trace_target.unsqueeze(0))) #(batch_size, n, window_size)
        up1 = self.upsample1(y1) #(batch_size, 2*n, window_size * 5)
        y2 = self.nonlinearity(self.conv2(up1)) #(batch_size, 2*n, window_size * 5)
        up2 = self.upsample2(y2) #(batch_size, 2*n, window_size * R)
        sensory = self.nonlinearity(self.conv3(sensory_input.unsqueeze(0)))
        merge = torch.cat((up2, sensory), 1) #(batch_size, 4*n, window_size * R)
        mu_neuron_voltage_latent = self.conv4(merge).squeeze(0) #(n, window_size * R)
        logvar_neuron_voltage_latent = self.conv5(merge).squeeze(0) #(n, window_size * R)
        # TODO: dependence of mu, std (TBD)
        std_neuron_voltage_latent = torch.exp(0.5*logvar_neuron_voltage_latent)
        # normal distribution
        eps = torch.randn_like(std_neuron_voltage_latent)
        # reparametrization
        sample_neuron_voltage_latent = mu_neuron_voltage_latent + eps * std_neuron_voltage_latent
        
        return mu_neuron_voltage_latent,logvar_neuron_voltage_latent,sample_neuron_voltage_latent
        # (n, window_size * R), (n, window_size * R), (n, window_size * R)

class WormNetCalcium(nn.Module):

    """ Network model
    Attributes: connectome constrained (weight initialization)
    decoder: q({F}|{X},{S})
    """

    def __init__(self, connectome, window_size, initialization, R, signs_c, encoder = nn.Linear, inference_network = nn.Linear, device = 'cuda', dt = 0.2, nonlinearity = F.relu, is_Training = True):
        super(WormNetCalcium, self).__init__()
        self.dt = dt
        self.cnctm_data = connectome

        cnctm_dict = {'sparsity_c': connectome.synapses_dict['chem_adj'],
                      'sparsity_e': connectome.synapses_dict['esym_adj'],
                      'signs_c': signs_c,
                      'magnitudes_c': connectome.synapses_dict['chem_weights'],
                      'magnitudes_e': connectome.synapses_dict['eassym_weights'],
                      'n': connectome.N}

        self.connectome = ConnectomeConstructor(cnctm_dict)
        self.sensory_mask = connectome.neuron_mask_dict['sensory']
        self.encoder = encoder
        self.inference_network = inference_network
        self.network_dynamics = cnctmnn.leaky_integrator_VAE(connectome = self.connectome,
                                                            dt = dt,
                                                            hidden_init_trainable = False,
                                                            bias_init = initialization['neuron_bias'],
                                                            tau_init = initialization['neuron_tau'],
                                                            nonlinearity = nonlinearity,
                                                            is_Training = is_Training)
         
        # voltage to calcium activation:
        # sigmoid(wc * X(t) + bc)
        self.voltage_to_calcium_activation_filter = cnctmnn.NeuronWiseAffine(in_features = connectome.N)
                                                        
        # calcium to fluorescence:
        # F(t) = wf * C(t) + bf + \epsilon, \epsilon = sigma_{\epsilon} * n_{\epsilon}, \epsilon ~ N(0,1)
        self.calcium_to_fluorescence_filter = cnctmnn.NeuronWiseLinear(in_features = connectome.N)

        self.loss = self.ELBO_loss
        self.logvar_fluorescence_trace_prob = Parameter(torch.rand(connectome.N,window_size * R))
        self.logvar_neuron_voltage_prob = Parameter(torch.rand(connectome.N,window_size * R))
        self.calcium_tau = Parameter(1e-2*torch.rand((1,connectome.N)))
        
    def forward(self, fluorescence_raw_target, fluorescence_full_target, odor_input, hidden_init = None):
        # fluorescence_raw_target: channel number: recorded neurons number, not in order
        # fluorescence_full_target: channel number: n (total neurons), in order
        # sensory input for sensory neurons (n, window_size * R)
        sensory_input = (self.encoder.forward(odor_input).T * self.sensory_mask).T
        #encoder output: mu, sigma, samples from inference network (n, window_size * R)
        mu_neuron_voltage_latent, logvar_neuron_voltage_latent,sample_neuron_voltage_latent = self.inference_network.forward(fluorescence_raw_target, sensory_input)
        # sample_neuron_voltage_latent (n, window_size * R)
        # hidden states: X(t) samples from inference network
        hidden_init = sample_neuron_voltage_latent
        
        # generative model: P(X(t)|X(t-1),S(t))~ N(\mu_{x(t)},{\sigma_x}^2)
        # \mu_{x(t)} = f(X(t-1),S(t)), f: leaky integration model (network_dynamics)
        # mu_neuron_activations: \mu_{x(t)}, recurrent_input: chem_in + eletric_in, sensory_input: S(t) (n, window_size * R)
        mu_neuron_voltage_prob, recurrent_in = self.network_dynamics.forward(sensory_input, hidden_init)
        
        # fluroscence model:
        # calcium_activation: sigmoid(Wf*X(t)+bf) (n, window_size * R)
        calcium_activation = self.voltage_to_calcium_activation_filter.forward(sample_neuron_voltage_latent)
        # mu_calcium_prob: C(t), t'dC(t)/dt + C(t) = sigmoid(Wf*X(t)+bf) (n, window_size * R)
        # implementation: exponential filter
        # init_calcium, initial condition: C(0)
        init_calcium = (fluorescence_full_target[:,0] - self.calcium_to_fluorescence_filter.shift)/self.calcium_to_fluorescence_filter.scale
        mu_calcium_prob = self.SCF_exponential(calcium_activation.unsqueeze(0), init_calcium)
        
        # calcium to fluorescence:
        # F(t) = \alpha * C(t) + \beta + \epsilon, \epsilon = sigma_{\epsilon} * n_{\epsilon}, \epsilon ~ N(0,1)
        # mu_fluorescence_trace_predict: \mu_f(t) (n, window_size * R)
        mu_fluorescence_trace_prob = self.calcium_to_fluorescence_filter.forward(torch.squeeze(mu_calcium_prob))
        
        return mu_neuron_voltage_prob, mu_fluorescence_trace_prob, mu_neuron_voltage_latent, logvar_neuron_voltage_latent, sample_neuron_voltage_latent, calcium_activation, mu_calcium_prob, recurrent_in, sensory_input
    
    def SCF_exponential(self, x, init_calcium):
        """modified version of Roman's explonential filter:
        Convolves x with exp(-alpha) = exp(-t/tau) for multiple cells.
        Args:
        x (torch.tensor): (batch_size, n_cells, R * window_size)
        alpha (torch.tensor): dt/tau (n_cells)
        kernel_size (int): size of kernel in time points
        Returns:
        conv (torch.tensor): (batch_size, n_cells,  R * window_size)
        """
        # batch_size = 1
        calcium_tau_clamp = self.calcium_tau.clamp(min=self.dt)
        alpha = self.dt/calcium_tau_clamp.squeeze(0)
        # here kernel size is chosen the same as the R * window_size
        timesteps = x.size()[2]
        kernel_size = timesteps
        t = torch.arange(0, kernel_size, step=1, dtype=torch.float32) # (kernel_size)
        kernel = torch.exp(-t*alpha[:, None]).flip(1)[:, None, :] # (n_cells, 1, kernel_size)
        conv = torch.nn.functional.conv1d(input=x,
                                          weight=kernel,
                                          groups=alpha.shape[0],
                                          padding=kernel_size-1)[:, :, :-kernel_size+1]
        step_t = torch.arange(0, timesteps, step=1, dtype=torch.float32)
        init_exp = torch.exp(-step_t*alpha[:, None]) # (n_cells, R * window_size)
        calcium_init_exp = (init_calcium[:,None] * init_exp)[None,:,:]
        
        return conv + calcium_init_exp # (batch_size, n_cells, window_size)
        
    def ELBO_loss(self, fluorescence_trace_target, missing_fluorescence_target, downsample_factor, mu_neuron_voltage_prob, mu_fluorescence_trace_prob, mu_neuron_voltage_latent, logvar_neuron_voltage_latent, sample_neuron_voltage_latent):
    
        """
        ELBO loss:
        L = -E_{z~q({X}|{F},{S})}[log(P({F}|{X},{S}))+log(P({X}|{S}))] + KL(q({X}|{F},{S})||Pn)
        """
        # reconstruction loss: log_likelihood
        # log(P({F}|{X},{S})), scalar average on (n, window_size * R)
        std_fluorescence_trace_prob = torch.exp(0.5*self.logvar_fluorescence_trace_prob)
        std_neuron_voltage_prob = torch.exp(0.5*self.logvar_neuron_voltage_prob)
        missing_ratio = torch.sum(missing_fluorescence_target)/(missing_fluorescence_target.shape[0]*missing_fluorescence_target.shape[1])
        recon_likelihood = -torch.mean((torch.distributions.normal.Normal(mu_fluorescence_trace_prob[:,::downsample_factor],std_fluorescence_trace_prob[:,::downsample_factor]).log_prob(fluorescence_trace_target))* (1 - missing_fluorescence_target))/(1 - missing_ratio)
        
        # trajectory probability distrbution: log_likelihood
        # log(P({X}|{S})), scalar average on (n, window_size * R)
        std_neuron_voltage_latent = torch.exp(0.5*logvar_neuron_voltage_latent)
        trajectory_log_likelihood = -torch.mean((torch.sum(torch.distributions.normal.Normal(mu_neuron_voltage_prob[:,1:],std_neuron_voltage_prob[:,1:]).log_prob(sample_neuron_voltage_latent[:,1:]),1) + torch.distributions.normal.Normal(mu_neuron_voltage_latent[:,0],std_neuron_voltage_latent[:,0]).log_prob(sample_neuron_voltage_latent[:,0]))/mu_neuron_voltage_prob.shape[1])
        
        # KL divergence: KL(q({X}|{F},{S})||Pn), Pn normal distribution
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2), logvar_neuron_voltage_latent (n, window_size * R)
        KLD_latent = -0.5 * torch.mean(1 + logvar_neuron_voltage_latent - mu_neuron_voltage_latent.pow(2) - logvar_neuron_voltage_latent.exp())
        loss = recon_likelihood + trajectory_log_likelihood + KLD_latent
        
        return loss, recon_likelihood, trajectory_log_likelihood, KLD_latent

    def regularized_MSEloss(self, x, target, missing_target, male_reg = 0, shared_reg = 0, weight_reg = 0):
        msefit = torch.mean((1-missing_target)*(x - target)**2) 
        reg_male =  male_reg*(torch.sum(torch.abs(x * self.cnctm_data.neuron_mask_dict['sex_spec'].repeat(x.shape[0],1))))  
        reg_shared = shared_reg*(torch.sum(torch.abs(x * self.cnctm_data.neuron_mask_dict['shared'].repeat(x.shape[0],1))))
        weight_decay = weight_reg*(torch.norm(self.network_dynamics.magnitudes_c) + torch.norm(self.network_dynamics.magnitudes_e))
        return msefit + reg_male + reg_shared + weight_decay

    def huber_MSEloss(self, x, target, missing_target, delta = 5,male_reg = 0, shared_reg = 0, weight_reg = 0):
        huber_loss = torch.mean((1-missing_target)*(x.clamp(max = delta) - target)**2 + delta*(torch.abs(x.clamp(min = delta)) - 0.5*delta))
        reg_male =  male_reg*(torch.sum(torch.abs(x * self.cnctm_data.neuron_mask_dict['sex_spec'].repeat(x.shape[0],1))))  
        reg_shared = shared_reg*(torch.sum(torch.abs(x * self.cnctm_data.neuron_mask_dict['shared'].repeat(x.shape[0],1))))
        weight_decay = weight_reg*(torch.norm(self.network_dynamics.magnitudes_c) + torch.norm(self.network_dynamics.magnitudes_e))
        return huber_loss + reg_male + reg_shared + weight_decay     
        


