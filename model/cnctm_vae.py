import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import cnctm.nn as cnctmnn
from cnctm.utils.data import ConnectomeConstructor
from model.loss import ELBO_loss
import pdb

class Worm_Sensory_Encoder(nn.Module):
    """Sensory encoder for odor inputs
    Attrbutes:
        inputs: odor inputs
        outputs: sensory inputs for sensory neurons
    
    Methods:
        linear layers, activation: ReLU
    """

    def __init__(self, n_input, n_output, nonlinearity = F.softplus):
        super(Worm_Sensory_Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, n_output)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        return self.nonlinearity(self.linear1(x.permute(0,2,1)))

class Worm_Inference_Network(nn.Module):
    """Inference network for wormvae
    Attributes:
        structure parameters of inference network;
        inputs: observed fluorescence trace + odor inputs
        outputs: approximated posterior distribution of voltage dynamics
    Methods:
        network: 1dconv , linear
        latent variables for all neurons:
        mu: mean of volatge posterior distribution
        logvar: log variance of volatge posterior distribution
        inference network: Q(v|f,o)
        v: posterir distribution for neuron volatge
        o: odor inputs
        f: fluorescence trace
    """

    def __init__(self, neuron_num, inputs_channels, channels_dict, kernel_size_dict, scale_factor_dict, nonlinearity = F.relu):
        # convolution without padding, expand inputs beforehand to be cropped
        super(Worm_Inference_Network, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = inputs_channels, out_channels = channels_dict['conv1'], kernel_size = kernel_size_dict['conv1'])
        self.conv2 = torch.nn.Conv1d(in_channels = channels_dict['conv1'], out_channels = channels_dict['conv2'], kernel_size = kernel_size_dict['conv2'])
        self.conv3 = torch.nn.Conv1d(in_channels = neuron_num, out_channels = channels_dict['conv3'], kernel_size = kernel_size_dict['conv3'])
        self.conv4 = torch.nn.Conv1d(in_channels = channels_dict['conv2'] + channels_dict['conv3'], out_channels = neuron_num, kernel_size = kernel_size_dict['conv4'])
        self.conv5 = torch.nn.Conv1d(in_channels = channels_dict['conv2'] + channels_dict['conv3'], out_channels = neuron_num, kernel_size = kernel_size_dict['conv5'])
        self.nonlinearity = nonlinearity
        self.bn1 = torch.nn.BatchNorm1d(channels_dict['conv1'])
        self.bn2 = torch.nn.BatchNorm1d(channels_dict['conv3'])
        self.upsample1 = torch.nn.Upsample(scale_factor= scale_factor_dict['upsample1'],mode = 'linear', align_corners=True)
        self.upsample2 = torch.nn.Upsample(scale_factor= scale_factor_dict['upsample2'],mode = 'linear', align_corners=True)
        self.eps = 1e-8
        
    def forward(self, fluorescence_trace_target, missing_target_mask, sensory_input, is_Training = True):

        inputs = torch.cat((fluorescence_trace_target, missing_target_mask),1)
        y1 = self.nonlinearity(self.conv1(inputs)) 
        up1 = self.upsample1(y1)
        y2 = self.nonlinearity(self.conv2(up1))
        up2 = self.upsample2(y2)
        sensory = self.nonlinearity(self.conv3(sensory_input))
        merge = torch.cat((up2, sensory), 1)
        voltage_latent_mu = self.conv4(merge)

        if is_Training == True:
            voltage_latent_std =  0.1 * F.softplus(self.conv5(merge)) + self.eps
        else:
            voltage_latent_std =  0.0 * F.softplus(self.conv5(merge)) 
            
        voltage_latent_std_train = 0.1 * F.softplus(self.conv5(merge)) + self.eps
        voltage_latent_logvar = torch.log(voltage_latent_std**2)
        noise = torch.randn_like(voltage_latent_std)

        # reparametrization
        voltage_latent_sample = voltage_latent_mu + noise * voltage_latent_std

        infer_posterior = {'voltage_latent_mu': voltage_latent_mu,
                           'voltage_latent_logvar': voltage_latent_logvar,
                           'voltage_latent_sample': voltage_latent_sample,
                           'voltage_latent_std_train': voltage_latent_std_train,
                           }

        return infer_posterior

class WormVAE(nn.Module):

    """ Network model
    Attributes:
        inputs: 
            observed fluorescence trace
            observed odor inputs
        outputs: 
            approximated posterior distribution of voltage dynamics
            prior distribution of voltage dynamics
            reconstructed flurorescence trace
            chemical synapstic input
            electrical synapstic input
            encoded sensory input
            calcium concentration
    Methods:
        sensory encoder: sensory_in(t) = MLP(odor_in(t))
        inference_network Q(v|f,o): approximated posterior distribution of voltage dynamics
        generative model P(v|o): prior distribution for voltage dynamics from sensory inputs
            network dynamics: tau dv(t)/dt + v(t) = chem_in(t) + elec_in(t) + v_rest + sensory_in(t) + std_v * noise_v(t)
        generative model P(f|v,o): map from voltage dynamics into caclum fluorescence trace
            voltage to calcium activation: tau'd[Ca](t)/dt + [Ca](t) = softplus(v(t))
            calcium to fluorescence: f(t) = wf * [Ca](t) + bf + std_f * noise_f(t)
    """

    def __init__(self, connectome, window_size, initialization, upsample_factor, signs_c, encoder, inference_network, device, dt, nonlinearity, model_type, constraint):

        super(WormVAE, self).__init__()
        self.dt = dt
        self.cnctm_data = connectome
        self.upsample_factor = upsample_factor

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
        self.T_init_steps = 10
        self.eps = 1e-8

        if model_type == 'current':
            self.network_dynamics = cnctmnn.leaky_integrator_current(connectome = self.connectome,
                                                                     dt = dt,
                                                                     hidden_init_trainable = False,
                                                                     bias_init = initialization['neuron_bias'],
                                                                     tau_init = initialization['neuron_tau'],
                                                                     nonlinearity = nonlinearity,
                                                                     constraint = constraint)

        if model_type == 'conductance':
            self.network_dynamics = cnctmnn.leaky_integrator_conductance(connectome = self.connectome,
                                                                         dt = dt,
                                                                         hidden_init_trainable = False,
                                                                         bias_init = initialization['neuron_bias'],
                                                                         tau_init = initialization['neuron_tau'],
                                                                         nonlinearity = nonlinearity,
                                                                         constraint = constraint)
                                                                                  
        

        self.voltage_to_calcium_activation_filter = cnctmnn.NeuronWiseAffine(in_features = connectome.N)
        self.calcium_to_fluorescence_filter = cnctmnn.NeuronWiseLinear(in_features = connectome.N)
        self.loss = ELBO_loss(connectome.N, window_size, self.upsample_factor)
        self.calcium_tau = Parameter(torch.rand(1))
        
    def forward(self, fluorescence_full_target, fluorescence_target, missing_target_mask, odor_input, is_Training = True):
        # sensory inputs encoded from odor inputs
        sensory_input_extend = (self.encoder.forward(odor_input) * self.sensory_mask).permute(0,2,1)
        
        # voltage posterior estimated from inference network
        infer_posterior = self.inference_network.forward(fluorescence_full_target, missing_target_mask, sensory_input_extend, is_Training = is_Training)
        crop_sensory = int((sensory_input_extend.shape[2] - infer_posterior['voltage_latent_mu'].shape[2])/2)
        sensory_input = sensory_input_extend[:,:,crop_sensory: sensory_input_extend.shape[2] - crop_sensory]
        hidden_init = infer_posterior['voltage_latent_sample']
        
        # voltage prior from network dynamics 
        prior_voltage_mu, chem_input, elec_input = self.network_dynamics.forward(sensory_input, hidden_init, is_Training = is_Training)
        
        # neurotransmitter concentration for calcium activation
        calcium_activation = self.voltage_to_calcium_activation_filter.forward(infer_posterior['voltage_latent_sample'])
        
        # calcium initial state estimation
        init_calcium_raw = (torch.mean(fluorescence_target[:,:,0:self.T_init_steps],dim=2) - self.calcium_to_fluorescence_filter.shift[None,:])/(self.calcium_to_fluorescence_filter.scale[None,:]+self.eps)
        init_calcium_raw = init_calcium_raw.detach()
        calcium_raw = (fluorescence_target - self.calcium_to_fluorescence_filter.shift[None,:,None])/(self.calcium_to_fluorescence_filter.scale[None,:,None]+self.eps)
        calcium_raw = calcium_raw.detach()
        
        # calcium concentration from explonential filter
        calcium_tau = self.calcium_tau * torch.ones_like(calcium_activation[:,:,0])
        pred_calcium_mu = self.SCF_exponential(calcium_activation/(calcium_tau[:,:,None]+self.eps), init_calcium_raw, calcium_tau)
       
        # fluroscence magnitude using affine transform
        pred_fluorescence_mu = self.calcium_to_fluorescence_filter.forward(pred_calcium_mu)

        outputs = infer_posterior
        outputs['pred_calcium_mu']=pred_calcium_mu
        outputs['pred_fluorescence_mu']=pred_fluorescence_mu
        outputs['prior_voltage_mu']=prior_voltage_mu
        outputs['chem_input']=chem_input
        outputs['elec_input']=elec_input
        outputs['sensory_input']=sensory_input
        outputs['calcium_activation']=calcium_activation

        return outputs
    
    def SCF_exponential(self, x, init_calcium, calcium_tau):
        """Explonential filter to generate calcium concentration from voltage.
        Convolves x with exp(-alpha) = exp(-t/tau) for multiple neurons.

        Args:
        x (torch.tensor): (batch_size, n_cells, R * window_size)
        alpha (torch.tensor): dt/tau (n_cells)
        kernel_size (int): size of kernel in time points
        Returns:
        conv (torch.tensor): (batch_size, n_cells,  R * window_size)
        """
        alpha = self.dt/(calcium_tau.squeeze(0))
        timesteps = x.size()[2]
        kernel_size = timesteps
        t = torch.arange(0, kernel_size, step=1, dtype=torch.float32)
        kernel = torch.exp(-t*alpha[:, None]).flip(1)[:, None, :]
        conv = torch.nn.functional.conv1d(input=x,
                                          weight=kernel,
                                          groups=alpha.shape[0],
                                          padding=kernel_size-1)[:, :, :-kernel_size+1]
        step_t = torch.arange(0, timesteps, step=1, dtype=torch.float32)
        init_exp = torch.exp(-step_t*alpha[:, None])
        calcium_init_exp = (init_calcium[:, :, None] * init_exp[None,:,:])
        return conv * self.dt + calcium_init_exp