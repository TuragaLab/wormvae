import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset
import glob
from os.path import dirname, join as pjoin
import scipy.io as sio

class Worm_Data_Loader(Dataset):
    def __init__(self, basepath) -> None:

        '''     
        A data preprocessing class for recorded neuron activity respect to olfactory odor inputs in each worm.
    
        Parameters:
            basepath: a path to the base directory containing relevant directory

        Attributes:
            activity_worms: list of dataframes containing calcium imaging data for each worm
            odor_worms: list of dataframes containing odor inputs for each worm
            neuron_names: list of neuron names
        '''
        # activity data attributes
        self.odor_channels = 3
        self.step = 0.25
        self.N_dataset = 21

        # mat file attributes
        N_cell = 189
        T = 960
        N_length = 109
        T_start = 160
        activity_datasets = np.zeros((self.N_dataset, N_cell, T))
        odor_datasets = np.zeros((self.N_dataset, self.odor_channels, T))
        
        mat_fname = pjoin(basepath, 'all_traces_Heads_new.mat')
        trace_variable = sio.loadmat(mat_fname)
        trace_arr = trace_variable['traces']
        is_L = trace_variable['is_L']
        neurons_name = trace_variable['neurons']
        stim_names = trace_variable["stim_names"]
        stimulate_seconds = trace_variable['stim_times']
        stims = trace_variable['stims']
        for idata in range(self.N_dataset):
            ineuron = 0
            for ifile in range(N_length):
                if trace_arr[ifile][0].shape[1] == 42:
                    data = trace_arr[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        activity_datasets[idata][ineuron][:] = np.nan
                    else:
                        activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                    data = trace_arr[ifile][0][0][idata + 21]
                    if data.shape[0] < 1:
                        activity_datasets[idata][ineuron][:] = np.nan
                    else:
                        activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                else:
                    data = trace_arr[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        activity_datasets[idata][ineuron][:] = np.nan
                    else:
                        activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
        # add baseline 2
        self.activity_worms = activity_datasets[:,:, T_start:] + 2
        
        neuron_names = []
        for ifile in range(N_length):
            if is_L[ifile][0][0].shape[0] == 42:
                neuron_names.append(neurons_name[ifile][0][0] + 'L')
                neuron_names.append(neurons_name[ifile][0][0] + 'R')
            else:
                neuron_names.append(neurons_name[ifile][0][0])
        self.neuron_names = neuron_names
        
        time = np.arange(start = 0, stop = T * self.step , step = self.step)
        self.odor_list = ['butanone','pentanedione','NaCL']
        for idata in range(self.N_dataset):
            for it_stimu in range(stimulate_seconds.shape[0]):
                tim1_ind = time>stimulate_seconds[it_stimu][0]
                tim2_ind = time<stimulate_seconds[it_stimu][1]
                odor_on = np.multiply(tim1_ind.astype(np.int),tim2_ind.astype(np.int))
                stim_odor = stims[idata][it_stimu] - 1
                odor_datasets[idata][stim_odor][:] = odor_on
                
        self.odor_worms = odor_datasets[:,:, T_start:]
    
    def process_trace(self,trace):
        '''
        Returns activity traces with normalization based on mean and standard devation.
        '''
        worm_trace = (trace - np.nanmean(trace))/np.nanstd(trace)
        return worm_trace

    def __process_activity(self):
        '''
        Returns a list of matrices corresponding to the data missing in the activity columns of the activity_worms dataframes and
        a matrix of the activity with NaNs replaced by 0's
        '''
        missing_data, activity_data = [],[]
        for worm in self.activity_worms:
            worm = (worm - worm.mean())/worm.std()
            act_matrix = worm.values
            missing_act = np.zeros(act_matrix.shape)
            missing_act[np.isnan(act_matrix)] = 1
            act_matrix[np.isnan(act_matrix)] = 0
            missing_data.append(missing_act)
            activity_data.append(act_matrix)
        return activity_data, missing_data
    
    def interpolate_odor(self, odor_worms, upsample_factor):
        '''
        Inputs:
            odor_worms: odor inputs of worm
            upsample_factor: interpolation upsampling factor
        
        Returns:
            list of Dataframes containing upsampled odor input data for each worm 
        '''
        B, N, T = odor_worms.shape
        upsampled_odor_worms = np.zeros((B,N,int(T*upsample_factor)+1))

        for worm_id in range(B):
            interp_matrix = np.zeros(shape = (N, int(T*upsample_factor)+1))
            for i in range(N):
                inter_points = np.linspace(0, T, int(T*upsample_factor)+1)
                interp_matrix[i,:] = np.interp(inter_points, range(T), odor_worms[worm_id][i,:])
            upsampled_odor_worms[worm_id,:,:] = interp_matrix
        return upsampled_odor_worms

def generate_target_mask(target_data):
    '''
    Returns the target with extra dimensions concatenated as zero columns and the
    missing data vector: (target, missing_target)
    '''
    full_missing_target = torch.zeros(size = (target_data.shape))
    full_missing_target[torch.isnan(target_data)] = 1
    full_target = target_data
    full_target[torch.isnan(target_data)] = 0
    return full_target, full_missing_target

class TimeSeriesDataloader(Dataset):
    def __init__(self, data_param_dict, data, window_size):
        super(TimeSeriesDataloader, self).__init__()
        self.data = data
        self.window_size = window_size
        self.max_index = self.data[0].shape[2] // self.window_size[0] - 2
        self.crop_target = int((data_param_dict['k1'] - 1 + (data_param_dict['k2'] - 1)/data_param_dict['up1_factor'] + (data_param_dict['k4'] - 1)/(data_param_dict['up1_factor'] * data_param_dict['up2_factor']))/2)
        self.crop_stim_feature = int((data_param_dict['k3'] - 1   + data_param_dict['k4'] - 1)/2)
        
    def __getitem__(self, index):
        return self.data[0][:,:, (index+1)*self.window_size[0] - self.crop_stim_feature:(index+2)*self.window_size[0] + self.crop_stim_feature], self.data[1][:,:, (index+1)*self.window_size[1] - self.crop_target:(index+2)*self.window_size[1] + self.crop_target], self.data[2][:,:, (index+1)*self.window_size[2] - self.crop_target:(index+2)*self.window_size[2] + self.crop_target], self.data[1][:, :, (index+1)*self.window_size[1]:(index+2)*self.window_size[1]], self.data[2][:, :, (index+1)*self.window_size[2]:(index+2)*self.window_size[2]]

    def __len__(self):
        return self.max_index

class SingleBatchDataloader(Dataset):
    def __init__(self, data, window_size):
        super(SingleBatchDataloader, self).__init__()
        self.data = data
        self.window_size = window_size
        self.total_max_index = self.data[0].shape[0] // self.window_size[0]
        self.current_index = 0
        self.max_index = self.current_index + 1
    
    def __getitem__(self, _ ):
        out0 = self.data[0][self.current_index*self.window_size[0]:(self.current_index+1)*self.window_size[0], :]
        out1 = self.data[1][self.current_index*self.window_size[1]:(self.current_index+1)*self.window_size[1], :]
        out2 = self.data[2][self.current_index*self.window_size[2]:(self.current_index+1)*self.window_size[2], :]
        self.current_index += 1
        if self.current_index >= self.total_max_index:
            self.max_index = 0
        return out0, out1, out2

    def __len__(self):
        return self.max_index

if __name__ == "__main__":
    worm_data_loader = Worm_Data_Loder('../of_data/')
    
