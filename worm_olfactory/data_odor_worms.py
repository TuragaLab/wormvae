import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset
import glob
from os.path import dirname, join as pjoin
import scipy.io as sio
import pdb

class Of_Data:
    def __init__(self, datapath, dataset = 'new') -> None:
        #TODO Make datafields tensors

        '''     
        A data preprocessing class for extracting neuron/synapse data relevant to worm olfactory behavior
    
        Parameters:
            basepath: a path to the base directory containing relevant directory

        Attributes:
            activity_worms: list of dataframes containing calcium imaging data for each worm
            behavior_worms: list of dataframes containing behavioral data for each worm (odor)
            missing_activity: list of matrices containing 1 in elements where data is missing
            masked_activity: list of matrices for each worm with activity concatenated with missing_activity
        '''
        # initialization and parameters designed

        #head data read
        N_dataset = 21
        N_cell = 189
        T = 960
        N_length = 109
        odor_channels = 3
        T_start = 160
        trace_datasets = np.zeros((N_dataset, N_cell, T))
        odor_datasets = np.zeros((N_dataset, odor_channels, T))
        name_list = []
        
        # .mat data load
        basepath = datapath
        mat_fname = pjoin(basepath, 'all_traces_Heads_new.mat')
        trace_variable = sio.loadmat(mat_fname)
        #trace_arr = trace_variable['norm_traces']
        trace_arr = trace_variable['traces']
        is_L = trace_variable['is_L']
        neurons_name = trace_variable['neurons']
        stim_names = trace_variable["stim_names"]
        stimulate_seconds = trace_variable['stim_times']
        stims = trace_variable['stims']
        # multiple trace datasets concatnate
        for idata in range(N_dataset):
            ineuron = 0
            for ifile in range(N_length):
                if trace_arr[ifile][0].shape[1] == 42:
                    data = trace_arr[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        trace_datasets[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                    data = trace_arr[ifile][0][0][idata + 21]
                    if data.shape[0] < 1:
                        trace_datasets[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                else:
                    data = trace_arr[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        trace_datasets[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
        # neural activity target
        self.activity_worms = trace_datasets[:,:, T_start:] + 2
        # normalization
        # self.activity_worms = self.process_trace(trace_datasets[:,:, T_start:])
                    
        # neuron name list (189 in total)
        name_list = []
        for ifile in range(N_length):
            if is_L[ifile][0][0].shape[0] == 42:
                name_list.append(neurons_name[ifile][0][0] + 'L')
                name_list.append(neurons_name[ifile][0][0] + 'R')
            else:
                name_list.append(neurons_name[ifile][0][0])
        self.activity_list = name_list
        
        step = 0.25
        time = np.arange(start = 0, stop = T * step , step = step)
        # odor list
        self.behavior_list = ['butanone','pentanedione','NaCL']
        # multiple odor datasets concatnate
        for idata in range(N_dataset):
            for it_stimu in range(stimulate_seconds.shape[0]):
                tim1_ind = time>stimulate_seconds[it_stimu][0]
                tim2_ind = time<stimulate_seconds[it_stimu][1]
                odor_on = np.multiply(tim1_ind.astype(np.int),tim2_ind.astype(np.int))
                stim_odor = stims[idata][it_stimu] - 1
                odor_datasets[idata][stim_odor][:] = odor_on
                
        self.behavior_worms = odor_datasets[:,:, T_start:]


        #Tails data

        N_dataset_1 = 21
        N_cell_1 = 42
        T = 960
        N_length_1 = 30
        odor_channels = 3
        T_start = 160
        trace_datasets_tails = np.zeros((N_dataset_1, N_cell_1, T))
        odor_datasets_tails = np.zeros((N_dataset_1, odor_channels, T))
        name_list_tails = []
        basepath = datapath

        mat_fname_1 = pjoin(basepath, 'all_traces_Tails_new.mat')
        #pdb.set_trace()
        trace_variable_1 = sio.loadmat(mat_fname_1)
        #pdb.set_trace()
        #trace_arr = trace_variable['norm_traces']
        trace_arr_1 = trace_variable_1['traces']
        is_L_1 = trace_variable_1['is_L']
        neurons_name_1 = trace_variable_1['neurons']
        stim_names_1 = trace_variable_1["stim_names"]
        stimulate_seconds_1 = trace_variable_1['stim_times']
        stims_1 = trace_variable_1['stims']
        # multiple trace datasets concatnate
        for idata in range(N_dataset_1):
            ineuron = 0
            for ifile in range(N_length_1):
                if trace_arr_1[ifile][0].shape[1] == 42:
                    data = trace_arr_1[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        trace_datasets_tails[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets_tails[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                    data = trace_arr_1[ifile][0][0][idata + 21]
                    if data.shape[0] < 1:
                        trace_datasets_tails[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets_tails[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                else:
                    data = trace_arr_1[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        trace_datasets_tails[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets_tails[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
        # neural activity target
        self.activity_worms_tails = trace_datasets_tails[:,:, T_start:] + 2
        # normalization
        # self.activity_worms = self.process_trace(trace_datasets[:,:, T_start:])
                    
        # neuron name list (42 in total)
        name_list_tails = []
        for ifile in range(N_length_1):
            if is_L_1[ifile][0][0].shape[0] == 42:
                name_list_tails.append(neurons_name_1[ifile][0][0] + 'L')
                name_list_tails.append(neurons_name_1[ifile][0][0] + 'R')
            else:
                name_list_tails.append(neurons_name_1[ifile][0][0])
        self.activity_list_tails = name_list_tails
        
        step = 0.25
        time = np.arange(start = 0, stop = T * step , step = step)
        # odor list
        self.behavior_list = ['butanone','pentanedione','NaCL']
        # multiple odor datasets concatnate
        for idata in range(N_dataset_1):
            for it_stimu in range(stimulate_seconds_1.shape[0]):
                tim1_ind = time>stimulate_seconds_1[it_stimu][0]
                tim2_ind = time<stimulate_seconds_1[it_stimu][1]
                odor_on = np.multiply(tim1_ind.astype(np.int),tim2_ind.astype(np.int))
                stim_odor = stims_1[idata][it_stimu] - 1
                odor_datasets_tails[idata][stim_odor][:] = odor_on
                
        self.behavior_worms_tails = odor_datasets_tails[:,:, T_start:]
    
    def process_trace(self,trace):
        #normalization based on mean, variance
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
    
    def interpolate_behavior(self, worm_list, dt):
        '''
        Parameters:
            worm_list - list of worm numbers
            dt - time step size
        
        Returns:
            list of Dataframes containing upsampled behavioral data for each worm 
        '''
        #pdb.set_trace()
        upsample_factor = 0.25/dt
        upsampled_worms = np.zeros((worm_list.shape[0],worm_list.shape[1],int(worm_list.shape[2]*upsample_factor)+1))
        D, T = worm_list.shape[1], worm_list.shape[2]
        for worm_id in range(worm_list.shape[0]):
            interp_matrix = np.zeros(shape = (D, int(T*upsample_factor)+1))
            for i in range(D):
                inter_points = np.linspace(0, T, int(T*upsample_factor)+1)
                interp_matrix[i,:] = np.interp(inter_points, range(T), self.behavior_worms[worm_id][i,:])
            upsampled_worms[worm_id,:,:] = interp_matrix
        #TBD: SIZE
        return upsampled_worms

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
    def __init__(self, data_param_dict, data, window_size): # trace.shape = (tpoints)
        super(TimeSeriesDataloader, self).__init__()
        self.data = data  # (dataset, n_neurons, T)
        self.window_size = window_size
        self.max_index = self.data[0].shape[2] // self.window_size[0] - 2
        self.crop_target = int((data_param_dict['k1'] - 1 + (data_param_dict['k2'] - 1)/data_param_dict['up1_factor'] + (data_param_dict['k4'] - 1)/(data_param_dict['up1_factor'] * data_param_dict['up2_factor']))/2)
        self.crop_stim_feature = int((data_param_dict['k3'] - 1   + data_param_dict['k4'] - 1)/2)
        

    def __getitem__(self, index):
        return self.data[0][:,:, (index+1)*self.window_size[0] - self.crop_stim_feature:(index+2)*self.window_size[0] + self.crop_stim_feature], self.data[1][:,:, (index+1)*self.window_size[1] - self.crop_target:(index+2)*self.window_size[1] + self.crop_target], self.data[2][:,:, (index+1)*self.window_size[2] - self.crop_target:(index+2)*self.window_size[2] + self.crop_target], self.data[1][:, :, (index+1)*self.window_size[1]:(index+2)*self.window_size[1]], self.data[2][:, :, (index+1)*self.window_size[2]:(index+2)*self.window_size[2]]

    def __len__(self):
        return self.max_index
        
class Ablation_TimeSeriesDataloader(Dataset):
    def __init__(self, data_param_dict, data, window_size): # trace.shape = (tpoints)
        super(Ablation_TimeSeriesDataloader, self).__init__()
        self.data = data  # (dataset, n_neurons, T)
        self.window_size = window_size
        self.max_index = self.data[0].shape[2] // self.window_size[0] - 2
        self.crop_target = int((data_param_dict['k1'] - 1 + (data_param_dict['k2'] - 1)/data_param_dict['up1_factor'] + (data_param_dict['k4'] - 1)/(data_param_dict['up1_factor'] * data_param_dict['up2_factor']))/2)
        self.crop_stim_feature = int((data_param_dict['k3'] - 1   + data_param_dict['k4'] - 1)/2)
    
    def __getitem__(self, index):
        input_window = self.data[0][:,:, (index+1)*self.window_size[0] - self.crop_stim_feature:(index+2)*self.window_size[0] + self.crop_stim_feature]
        target_window_padding = self.data[1][:,:, (index+1)*self.window_size[1] - self.crop_target:(index+2)*self.window_size[1] + self.crop_target]
        missing_window_padding = self.data[2][:,:, (index+1)*self.window_size[2] - self.crop_target:(index+2)*self.window_size[2] + self.crop_target]
        target_window = self.data[1][:, :, (index+1)*self.window_size[1]:(index+2)*self.window_size[1]]
        missing_window = self.data[2][:, :, (index+1)*self.window_size[2]:(index+2)*self.window_size[2]]
        ablation_target_padding = self.data[3][:,:, (index+1)*self.window_size[1] - self.crop_target:(index+2)*self.window_size[1] + self.crop_target]
        ablation_missing_target_padding = self.data[4][:,:, (index+1)*self.window_size[2] - self.crop_target:(index+2)*self.window_size[2] + self.crop_target]
        return input_window, target_window_padding, missing_window_padding, target_window, missing_window, ablation_target_padding, ablation_missing_target_padding

    def __len__(self):
        return self.max_index

class SingleBatchDataloader(Dataset):
    def __init__(self, data, window_size): # trace.shape = (tpoints)
        super(SingleBatchDataloader, self).__init__()
        self.data = data  # (timesteps, n_neurons)
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
    worm_data = Of_Data('../of_data/')
    
