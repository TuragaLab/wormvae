import sys
from pathlib import Path
import torch
from torch.optim.lr_scheduler import StepLR
import logging
import datetime, time
from model import *
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import pdb

def train(data_loader, network, model_type, constraint, lr_params, savepath):
    '''WormVAE training procedure
    Attributes:
        input:
            data loader
            network
            model type (synapse type)
            constraint (connectome constrained: weight v.s. sparsity v.s. unconstrained)
            learning hyperparameter
        outputs:
            loss trajectory
            model checkpoint
    '''
    params = list(network.parameters())
    for name, param in network.named_parameters():
        if param.requires_grad:
            logging.info(name)
            logging.info(param.data)
    
    optimizer = torch.optim.Adam(network.parameters(),lr = lr_params['lr'])

    scheduler = StepLR(optimizer, step_size = lr_params['step_size'], gamma = lr_params['gamma'])
    loss_list, recon_loss_list, KLD_list = [], [], []
    for epoch in range(lr_params['epochs']):
        iter, loss_sum, recon_loss_sum, KLD_sum  = 0, 0, 0, 0
        for odor_input_window, target_window_padding, missing_window_padding, target_window, missing_window in data_loader:
            optimizer.zero_grad()

            loss, recon_likelihood, KLD_latent = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            batch_size = odor_input_window.shape[1]
            
            # train loop
            for iworm in range(batch_size):

                # worm vae forward
                outputs = network.forward(fluorescence_full_target = target_window_padding[:,iworm,:,:],
                                         fluorescence_target = target_window[:,iworm,:,:],
                                         missing_target_mask = missing_window_padding[:,iworm,:,:],
                                         odor_input = odor_input_window[:,iworm,:,:],
                                         is_Training = True)
                # loss calculation
                loss_minibatch, recon_likelihood_minibatch, KLD_latent_minibatch = network.loss(
                    fluorescence_trace_target = target_window[:,iworm,:,:],
                    missing_fluorescence_target =  missing_window[:,iworm,:,:],
                    outputs = outputs,
                    recon_weight = 1,
                    KLD_weight = 1)
                
                # back propagation
                loss += loss_minibatch
                recon_likelihood += recon_likelihood_minibatch
                KLD_latent += KLD_latent_minibatch
                loss_minibatch.backward()
                torch.nn.utils.clip_grad_value_(params, lr_params['grad_clip'])

            # loss monitor   
            loss /= batch_size
            recon_likelihood /= batch_size
            KLD_latent /= batch_size

            loss_item = loss.detach().item()
            recon_loss_item = recon_likelihood.detach().item()
            KLD_item = KLD_latent.detach().item()
            
            loss_sum += loss_item
            recon_loss_sum += recon_loss_item
            KLD_sum += KLD_item
            
            # optimizer update
            optimizer.step()

            # time scales, initializations, and synaptic weights must all be positive
            with torch.no_grad():
                if model_type == 'conductance':
                    network.network_dynamics.magnitudes_c.data.clamp_(min = 0)
                if constraint == 'weight':
                    network.network_dynamics.magnitude_scaling_factor_chem.data.clamp_(min = 0)
                    network.network_dynamics.magnitude_scaling_factor_elec.data.clamp_(min = 0)
                network.network_dynamics.magnitudes_e.data.clamp_(min = 0)
                network.calcium_to_fluorescence_filter.scale.data.clamp_(min = lr_params['eps'])
                network.network_dynamics.tau.data.clamp_(min = network.dt)
                network.calcium_tau.data.clamp_(min = network.dt)

            iter += 1
            print('[{}] epoch: {}, iter: {},  avg loss: {},  avg recon_loss: {},  avg KLD: {}'.format(datetime.datetime.now(),epoch, iter, loss_item, recon_loss_item, KLD_item))
            logging.info('[{}] epoch: {}, iter: {},  avg loss: {},  avg recon_loss: {},  avg KLD: {}'.format(datetime.datetime.now(),epoch, iter, loss_item, recon_loss_item, KLD_item))
            
        epoch_loss = loss_sum/iter
        epoch_recon_loss = recon_loss_sum/iter
        epoch_KLD = KLD_sum/iter
        
        print('[{}] epoch: {},  avg loss: {},  avg recon loss: {},  avg KLD: {}'.format(datetime.datetime.now(), epoch, epoch_loss, epoch_recon_loss, epoch_KLD))
        logging.info('[{}] epoch: {},  avg loss: {},  avg recon loss: {},  avg KLD: {}'.format(datetime.datetime.now(), epoch, epoch_loss, epoch_recon_loss, epoch_KLD))
        
        loss_list.append(epoch_loss)
        recon_loss_list.append(epoch_recon_loss)
        KLD_list.append(epoch_KLD)
        
        if (epoch + 1) % lr_params['ckpt_save_freq'] == 0:
            torch.save(network.state_dict(), savepath + f'epoch_{epoch}.pt')
        scheduler.step()
    return loss_list, recon_loss_list, KLD_list