import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import cnctm.nn as cnctmnn
from cnctm.utils.data import ConnectomeConstructor
import pdb
#from worm_olfactory.losses_worms_AE import ELBO_loss
from worm_olfactory.losses_worms import ELBO_loss

class Worm_Sensory_Encoder(nn.Module):
    """sensory encoder for odor inputs
    Attrbutes:
    inputs: odor inputs
    outputs: sensory inputs for sensory neurons
    
    Methods:
    linear layers, activation: ReLU
    """
    # odor_inputs: after interpolation: (batch, odor_channel, T*R)
    # sensory_inputs (output): (batch, n, T*R)

    def __init__(self, n_input, n_output, nonlinearity = F.softplus):
        super(Worm_Sensory_Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, n_output)
        self.nonlinearity = nonlinearity
        #self.linear2 = torch.nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.nonlinearity(self.linear1(x.permute(0,2,1)))
        # output: (batch, T, n)

class Worm_Inference_Network(nn.Module):
    """ inference network for worm_vae_olfactory
    Attributes:
        structure parameters of encoder;
        inputs: fluorescence trace + sensory inputs
        outputs: latent variables
        R: time interpolation rate
    Methods:
        network: 1dconv , linear
        latent variables for all neurons:  mu:neuron_num * window_size, sigma: neuron_num * window_size
        encoder: q({X}|{F},{S})
        {S}: sensory inputs
        {X}: latent variables for all neurons
        {F}: fluorescence trace
    """

    def __init__(self, n, inputs_channels, channels_dict, kernel_size_dict, scale_factor_dict, nonlinearity = F.relu):
        # neuron num: n, window_size: T
        # conv without padding: 
        # fluroscence input size should be  T + (k1-1) + (k2-1)/up1 + (k4-1)/(up1 * up2), eg. k1: kernel_size_conv1
        # odor (sensory) input size should be T + (k3-1) + (k4-1)
        # target T size should be T 
        # k4 == k5
        # TODO: missing data mask
        super(Worm_Inference_Network, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = inputs_channels, out_channels = channels_dict['conv1'], kernel_size = kernel_size_dict['conv1'])
        self.conv2 = torch.nn.Conv1d(in_channels = channels_dict['conv1'], out_channels = channels_dict['conv2'], kernel_size = kernel_size_dict['conv2'])
        self.conv3 = torch.nn.Conv1d(in_channels = n, out_channels = channels_dict['conv3'], kernel_size = kernel_size_dict['conv3'])
        self.conv4 = torch.nn.Conv1d(in_channels = channels_dict['conv2'] + channels_dict['conv3'], out_channels = n, kernel_size = kernel_size_dict['conv4'])
        self.conv5 = torch.nn.Conv1d(in_channels = channels_dict['conv2'] + channels_dict['conv3'], out_channels = n, kernel_size = kernel_size_dict['conv5'])
        self.nonlinearity = nonlinearity
        self.bn1 = torch.nn.BatchNorm1d(channels_dict['conv1'])
        self.bn2 = torch.nn.BatchNorm1d(channels_dict['conv3'])
        # R = acqusition_dt/ simulation_dt
        # R =  upsample1 * upsample2
        self.upsample1 = torch.nn.Upsample(scale_factor= scale_factor_dict['upsample1'],mode = 'linear', align_corners=True)
        self.upsample2 = torch.nn.Upsample(scale_factor= scale_factor_dict['upsample2'],mode = 'linear', align_corners=True)
        
    def forward(self, fluorescence_trace_target, missing_target_mask, sensory_input):
        # inputs:
        # fluorescence_trace_target (n, T_F)
        # sensory_input (n, T_S)
        # batch_size = 1
        inputs = torch.cat((fluorescence_trace_target, missing_target_mask),1)
        y1 = self.nonlinearity(self.conv1(inputs)) #y1: (batch_size, channels_number_conv1, T_1)
        #y1 = self.bn1(y1)
        up1 = self.upsample1(y1) #up1: (batch_size, channels_number_conv1, T_up1)
        y2 = self.nonlinearity(self.conv2(up1)) #y2: (batch_size, channels_number_conv2, T_2)
        up2 = self.upsample2(y2) #up2: (batch_size, channels_number_conv2, T_up2)
        sensory = self.nonlinearity(self.conv3(sensory_input)) #sensory: (batch_size, channels_number_conv3, T_3)
        #sensory = self.bn2(sensory)
        merge = torch.cat((up2, sensory), 1) #merge: (batch_size, channels_number_conv2 + channels_number_conv3, T_3)
        mu_neuron_voltage_latent = self.conv4(merge) #(batch_size, n, T * R)
        #logvar_neuron_voltage_latent = self.conv5(merge) #(batch_size, n, T* R)
        #std_neuron_voltage_latent = torch.exp(0.5*logvar_neuron_voltage_latent) #(batch_size, n, T* R)
        #std_neuron_voltage_latent = 1 * F.softplus(self.conv5(merge))
        #std_neuron_voltage_latent = 0.1 * F.sigmoid(self.conv5(merge))
        #std_neuron_voltage_latent = 0.01 * F.softplus(self.conv5(merge))
        std_neuron_voltage_latent = 0.1 * F.softplus(self.conv5(merge))
        std_neuron_voltage_latent_train = 0.1 * F.softplus(self.conv5(merge))
        logvar_neuron_voltage_latent = torch.log(std_neuron_voltage_latent**2) #(batch_size, n, T* R)
        #logvar_neuron_voltage_latent = torch.zeros_like(std_neuron_voltage_latent) #(batch_size, n, T* R)
        # normal distribution
        eps = torch.randn_like(std_neuron_voltage_latent)
        # TODO: correlated posterior, RNN

        # steps to replace reparameterization 
        #posterior_sample = Normal(a, b).rsample()
        #posterior_sample.backward()

        # reparametrization
        sample_neuron_voltage_latent = mu_neuron_voltage_latent + eps * std_neuron_voltage_latent
        return mu_neuron_voltage_latent,logvar_neuron_voltage_latent,sample_neuron_voltage_latent, std_neuron_voltage_latent_train
        # (batch_size, n, T * R), (batch_size, n, T * R), (batch_size, n, T * R)

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
        self.network_dynamics = cnctmnn.leaky_integrator_VAE_worms(connectome = self.connectome,
                                                                   dt = dt,
                                                                   hidden_init_trainable = False,
                                                                   bias_init = initialization['neuron_bias'],
                                                                   tau_init = initialization['neuron_tau'],
                                                                   nonlinearity = nonlinearity,
                                                                   is_Training = is_Training)
         
        # voltage to calcium activation:
        # nolinear(wc * X(t) + bc)
        self.voltage_to_calcium_activation_filter = cnctmnn.NeuronWiseAffine(in_features = connectome.N)
                                                        
        # calcium to fluorescence:
        # F(t) = wf * C(t) + bf + \epsilon, \epsilon = sigma_{\epsilon} * n_{\epsilon}, \epsilon ~ N(0,1)
        self.calcium_to_fluorescence_filter = cnctmnn.NeuronWiseLinear(in_features = connectome.N)

        self.loss = ELBO_loss(connectome.N, window_size, R)
        #self.calcium_tau = Parameter(torch.rand((1,connectome.N)))
        self.calcium_tau = Parameter(torch.rand(1))
        
    def forward(self, fluorescence_full_target, missing_target_mask, odor_input, hidden_init = None):
        # fluorescence_full_target: (batch, n, T * R), channel number: n (total neurons), in order
        # sensory input for sensory neurons (batch, n, T * R)
        sensory_input_extend = (self.encoder.forward(odor_input) * self.sensory_mask).permute(0,2,1)
        #encoder output: mu, sigma, samples from inference network (batch, n, window_size * R)
        mu_neuron_voltage_latent, logvar_neuron_voltage_latent,sample_neuron_voltage_latent, std_neuron_voltage_latent_train = self.inference_network.forward(fluorescence_full_target, missing_target_mask, sensory_input_extend)
        crop_sensory = int((sensory_input_extend.shape[2] - mu_neuron_voltage_latent.shape[2])/2)
        sensory_input = sensory_input_extend[:,:,crop_sensory: sensory_input_extend.shape[2] - crop_sensory]
        # sample_neuron_voltage_latent (batch, n, T * R)
        # hidden states: X(t) samples from inference network
        hidden_init = sample_neuron_voltage_latent
        
        # generative model: P(X(t)|X(t-1),S(t))~ N(\mu_{x(t)},{\sigma_x}^2)
        # \mu_{x(t)} = f(X(t-1),S(t)), f: leaky integration model (network_dynamics)
        # mu_neuron_activations: \mu_{x(t)}, recurrent_input: chem_in + eletric_in, sensory_input: S(t) (batch, n, T * R)
        mu_neuron_voltage_prob, recurrent_in = self.network_dynamics.forward(sensory_input, hidden_init)
        # mu_neuron_voltage_prob: (batch, n, T * R)
        
        # fluroscence model:
        # calcium_activation: nonlinear(Wf*X(t)+bf) (batch, n, T * R)
        
        #calcium_activation = self.voltage_to_calcium_activation_filter.forward(sensory_input)
        calcium_activation = self.voltage_to_calcium_activation_filter.forward(sample_neuron_voltage_latent)
        # calcium_activation = sample_neuron_voltage_latent
        # mu_calcium_prob: C(t), t'dC(t)/dt + C(t) = sigmoid(Wf*X(t)+bf) (batch, n, T * R)
        # implementation: exponential filter
        # init_calcium, initial condition: C(0)
        #init_calcium = (fluorescence_full_target[:,:,0] - self.calcium_to_fluorescence_filter.shift[None,:])/self.calcium_to_fluorescence_filter.scale[None,:]
        
        # init_calcium, initial condition: C(0) inverse transform fluroscence by multiple steps
        T_init_steps = 10
        #pdb.set_trace()
        init_calcium = (torch.mean(fluorescence_full_target[:,:,0:T_init_steps],dim=2) - self.calcium_to_fluorescence_filter.shift[None,:])/self.calcium_to_fluorescence_filter.scale[None,:]

        #mu_calcium_prob = self.SCF_exponential(calcium_activation, init_calcium, self.calcium_tau)
        #mu_calcium_prob = self.SCF_exponential(calcium_activation/self.calcium_tau[:,:,None], init_calcium, self.calcium_tau)
        
        #same calicum tau for all neurons
        calcium_tau = self.calcium_tau * torch.ones_like(init_calcium)
        mu_calcium_prob = self.SCF_exponential(calcium_activation/calcium_tau[:,:,None], init_calcium, calcium_tau)
        # mu_calcium_prob: (batch, n, T * R)
        
        # calcium to fluorescence:
        # F(t) = \alpha * C(t) + \beta + \epsilon, \epsilon = sigma_{\epsilon} * n_{\epsilon}, \epsilon ~ N(0,1)
        # mu_fluorescence_trace_predict: \mu_f(t) (batch, n, T * R)
        mu_fluorescence_trace_prob = self.calcium_to_fluorescence_filter.forward(mu_calcium_prob)
        
        return mu_neuron_voltage_prob, mu_fluorescence_trace_prob, mu_neuron_voltage_latent, logvar_neuron_voltage_latent, sample_neuron_voltage_latent, std_neuron_voltage_latent_train, calcium_activation, mu_calcium_prob, recurrent_in, sensory_input
    
    def SCF_exponential(self, x, init_calcium, calcium_tau):
        """modified version of Roman's explonential filter:
        Convolves x with exp(-alpha) = exp(-t/tau) for multiple cells.
        Args:
        x (torch.tensor): (batch_size, n_cells, R * window_size)
        alpha (torch.tensor): dt/tau (n_cells)
        kernel_size (int): size of kernel in time points
        Returns:
        conv (torch.tensor): (batch_size, n_cells,  R * window_size)
        """
        alpha = self.dt/(calcium_tau.squeeze(0))
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
        init_exp = torch.exp(-step_t*alpha[:, None]) # (n_cells, R * T)
        calcium_init_exp = (init_calcium[:, :, None] * init_exp[None,:,:])
        #TBD
        
        return conv * self.dt + calcium_init_exp # (batch_size, n_cells, T)
        #return conv
        
        


