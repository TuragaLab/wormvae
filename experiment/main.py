import sys
from pathlib import Path
import torch
import logging
sys.path.append('../')
from model import *
from model import data_loader
import torch.nn.functional as F
import pickle
import argparse
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()
parser.add_argument("--neuron_holdout", default = [], help="neuron holdout list")
parser.add_argument("--train_worm", default = [1], help="train worm list")
parser.add_argument("--model_type", default = 'conductance', help="synapse model type")
parser.add_argument("--constraint", default = 'weight', help="connectome constraint for synapse weight")
parser.add_argument("--random_init_index", default = 0, help="model random initialize index")
args = parser.parse_args()

exp_name =  f'worm_vae_neuron_held_out_{args.neuron_holdout}_train_worm_{args.train_worm}_{args.model_type}_{args.constraint}_rand_{str(args.random_init_index)}_'

logging.basicConfig(filename = 'logs/'+exp_name+'logging.log',level=logging.DEBUG,format='%(message)s')

print('args.neuron_holdout:',str(args.neuron_holdout))
print('train_worm:',str(args.train_worm))
print('model_type:',str(args.model_type))
print('args.constraint:',str(args.constraint))
print('random_init_index:',str(args.random_init_index))

activity_path = '../data/worm_activity/'
cnctm_path = '../data/worm_connectivity/'
savepath = 'checkpoints/'+exp_name+'_checkpoint_'

worm_data_loader = Worm_Data_Loader(activity_path)
connectome_data = WhiteConnectomeData(cnctm_path, 'cuda')
num_train_datasets = len(args.train_worm)
stim_channels = worm_data_loader.odor_channels
acquisition_dt = worm_data_loader.step
N = connectome_data.N

# Simulation hyparameters
upsample_factor = 40
dt = acquisition_dt/upsample_factor
window_size = 30

# Training hyperparameters
device = 'cuda'
lr_params = {}
lr_params['lr'] = 3e-4
lr_params['step_size'] = 50
lr_params['epochs'] = 300
lr_params['gamma'] = 0.5
lr_params['grad_clip'] = 1.0
lr_params['eps'] = 1e-8
lr_params['ckpt_save_freq'] = 100

# Inference network
# network architecture hyperparameters 
inference_net_params = {'k1': 11, 'k2': 21, 'k3': 21, 'k4': 41, 'k5': 41, 'up1_factor': 4, 'up2_factor': 10}
channels_dict = {'conv1': 2 * N,'conv2': 2 * N,'conv3': 2 * N}
kernel_size_dict = {'conv1': inference_net_params['k1'],
                    'conv2': inference_net_params['k2'],
                    'conv3': inference_net_params['k3'],
                    'conv4': inference_net_params['k4'],
                    'conv5': inference_net_params['k5']}
scale_factor_dict = {'upsample1': inference_net_params['up1_factor'], 'upsample2': inference_net_params['up2_factor']}
inference_network = Worm_Inference_Network(
    neuron_num = N,
    inputs_channels = 2 * N,
    channels_dict = channels_dict,
    kernel_size_dict = kernel_size_dict,
    scale_factor_dict = scale_factor_dict,
    nonlinearity = F.relu)

# Sensory encoder
encoder = Worm_Sensory_Encoder(stim_channels,N)

# Worm VAE : inference network + generative model
initialization = {'neuron_bias': 1e-3*torch.rand((1, N)).to(device),
                  'neuron_tau': 1e-2*torch.rand((1, N)).to(device),
                  'calcium_bias': 1e-2*torch.rand((1, N)).to(device),
                  'voltage_to_calcium_nonlinearity': F.softplus}

signs_c = torch.empty(N,N).uniform_(-0.01,0.01)
network = WormVAE(
    connectome = connectome_data,
    window_size = window_size,
    initialization = initialization,
    upsample_factor = upsample_factor,
    signs_c = signs_c,
    encoder = encoder,
    inference_network = inference_network,
    device = device,
    dt = dt,
    nonlinearity = F.leaky_relu,
    model_type = args.model_type,
    constraint =args.constraint)

network.to(device)

# Train data loader
upsampled_df_list = worm_data_loader.interpolate_odor(worm_data_loader.odor_worms[args.train_worm],upsample_factor)
stim_features = torch.from_numpy(upsampled_df_list).to(torch.float32).cuda()
target_raw = torch.from_numpy(worm_data_loader.activity_worms[args.train_worm]).to(torch.float32).cuda()
full_target = connectome_data.make_full_connectome_activity(
    activity_data = target_raw,
    activity_neuron_list = worm_data_loader.neuron_names,
    holdout_list = args.neuron_holdout)
full_target = full_target.to(torch.float32).cuda()
target, missing_target = data_loader.generate_target_mask(full_target)
train_dataset = data_loader.TimeSeriesDataloader(data_param_dict = inference_net_params,
                                     data = [stim_features, target, missing_target],
                                     window_size = [int(window_size * upsample_factor), window_size, window_size])
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True, drop_last = False)

# Model train
loss_list, recon_loss_list, KLD_list = train(data_loader = trainloader,
                                             network = network,
                                             model_type = args.model_type,
                                             constraint =args.constraint,
                                             lr_params = lr_params,
                                             savepath = savepath)

loss_dict = {'loss': loss_list, 'recon_loss': recon_loss_list, 'KLD': KLD_list}
with open('loss_trajectories/' + exp_name+'_loss.pickle', 'wb') as handle:
    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

