import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import cnctm.nn as cnctmnn
from cnctm.utils.data import ConnectomeConstructor
import pdb
from worm_olfactory.losses import ELBO_loss

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
        latent variables for all neurons:  mu:neuron_num * window_size, sigma: neuron_num * window_size
        encoder: q({X}|{F},{S})
        {S}: sensory inputs
        {X}: latent variables for all neurons
        {F}: fluorescence trace
    """

    def __init__(self, n, recorded_n, channels_dict, kernel_size_dict, scale_factor_dict, nonlinearity = F.relu):
        # neuron num: n, window_size: T
        # conv without padding: 
        # fluroscence input size should be  T + (k1-1) + (k2-1)/up1 + (k4-1)/(up1 * up2), eg. k1: kernel_size_conv1
        # odor (sensory) input size should be T + (k3-1) + (k4-1)
        # target T size should be T 
        # k4 == k5
        # TODO: missing data mask
        super(Worm_Inference_Network, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = recorded_n, out_channels = channels_dict['conv1'], kernel_size = kernel_size_dict['conv1'])
        self.conv2 = torch.nn.Conv1d(in_channels = channels_dict['conv1'], out_channels = channels_dict['conv2'], kernel_size = kernel_size_dict['conv2'])
        self.conv3 = torch.nn.Conv1d(in_channels = n, out_channels = channels_dict['conv3'], kernel_size = kernel_size_dict['conv3'])
        self.conv4 = torch.nn.Conv1d(in_channels = channels_dict['conv2'] + channels_dict['conv3'], out_channels = n, kernel_size = kernel_size_dict['conv4'])
        self.conv5 = torch.nn.Conv1d(in_channels = channels_dict['conv2'] + channels_dict['conv3'], out_channels = n, kernel_size = kernel_size_dict['conv5'])
        self.nonlinearity = nonlinearity

        # R = acqusition_dt/ simulation_dt
        # R =  upsample1 * upsample2
        self.upsample1 = torch.nn.Upsample(scale_factor= scale_factor_dict['upsample1'],mode = 'linear', align_corners=True)
        self.upsample2 = torch.nn.Upsample(scale_factor= scale_factor_dict['upsample2'],mode = 'linear', align_corners=True)
        
    def forward(self, fluorescence_trace_target, sensory_input):
        # inputs:
        # fluorescence_trace_target (n, T_F)
        # sensory_input (n, T_S)
        # batch_size = 1
        y1 = self.nonlinearity(self.conv1(fluorescence_trace_target.unsqueeze(0))) #(batch_size, channels_number_conv1, T_1)
        up1 = self.upsample1(y1) #(batch_size, channels_number_conv1, T_up1)
        y2 = self.nonlinearity(self.conv2(up1)) #(batch_size, channels_number_conv2, T_2)
        up2 = self.upsample2(y2) #(batch_size, channels_number_conv2, T_up2)
        sensory = self.nonlinearity(self.conv3(sensory_input.unsqueeze(0))) #(batch_size, channels_number_conv3, T_3)
        merge = torch.cat((up2, sensory), 1) #(batch_size, channels_number_conv2 + channels_number_conv3, T_3)
        mu_neuron_voltage_latent = self.conv4(merge).squeeze(0) #(n, T * R)
        logvar_neuron_voltage_latent = self.conv5(merge).squeeze(0) #(n, T* R)
        std_neuron_voltage_latent = torch.exp(0.5*logvar_neuron_voltage_latent)
        #pdb.set_trace()
        # normal distribution
        eps = torch.randn_like(std_neuron_voltage_latent)
        # TODO: correlated posterior, RNN

        # steps to replace reparameterization 
        #posterior_sample = Normal(a, b).rsample()
        #posterior_sample.backward()

        # reparametrization
        sample_neuron_voltage_latent = mu_neuron_voltage_latent + eps * std_neuron_voltage_latent
        
        return mu_neuron_voltage_latent,logvar_neuron_voltage_latent,sample_neuron_voltage_latent
        # (n, T * R), (n, T * R), (n, T * R)

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

        self.loss = ELBO_loss(connectome.N, window_size, R)
        self.calcium_tau = Parameter(1e-2*torch.rand((1,connectome.N)))
        
    def forward(self, fluorescence_raw_target, fluorescence_full_target, odor_input, hidden_init = None):
        # fluorescence_raw_target: channel number: recorded neurons number, not in order
        # fluorescence_full_target: channel number: n (total neurons), in order
        # sensory input for sensory neurons (n, window_size * R)
        sensory_input_extend = (self.encoder.forward(odor_input).T * self.sensory_mask).T
        #encoder output: mu, sigma, samples from inference network (n, window_size * R)
        mu_neuron_voltage_latent, logvar_neuron_voltage_latent,sample_neuron_voltage_latent = self.inference_network.forward(fluorescence_raw_target, sensory_input_extend)
        crop_sensory = int((sensory_input_extend.shape[1] - mu_neuron_voltage_latent.shape[1])/2)
        sensory_input = sensory_input_extend[:,crop_sensory: sensory_input_extend.shape[1] - crop_sensory]
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
        alpha = self.dt/(self.calcium_tau.squeeze(0))
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
        
        


