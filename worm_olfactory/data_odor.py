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
    def __init__(self, datapath, foldername,dataset = 'new') -> None:
        #TODO Make datafields tensors

        '''     
        A data preprocessing class for extracting neuron/synapse data relevant to worm olfactory behavior
    
        Parameters:
            basepath: a path to the base directory containing relevant directory
            net: string path from basepath to synaptic data
            neurons: string path from basepath to neuron transmitter and annotation data
            calcium: string path from basepath to folder containing calcium imaging data
            neuron_list = None: list of neurons that we are interested in modeling. Defaults to     
                            all the neurons in the network

        Attributes:
            activity_worms: list of dataframes containing calcium imaging data for each worm
            behavior_worms: list of dataframes containing behavioral data for each worm
            missing_activity: list of matrices containing 1 in elements where data is missing
            missng_behavior: list of matrices containing 1 in elemenets where data is missing
            masked_activity: list of matrices for each worm with activity concatenated with missing_activity
            masked_behavior: list of matrices for each worm with behavior concatenated with missing_behavior 
        '''
        basepath = datapath
        
        #remove the the initial segment in the data
        clip_fraction = 1/6
        
        total_experiment = 1
        for if_name in range(total_experiment):
            data_dir = pjoin(basepath,'OH16230_Raw_Data', '%s' %(foldername))
            mat_fname = pjoin(data_dir, 'traces.mat')
            id_fname = pjoin(data_dir, 'neuron_id.mat')
            pos_fname = pjoin(data_dir, 'positions.mat')
            pos_variable = sio.loadmat(pos_fname)
            id_variable = sio.loadmat(id_fname)
            trace_variable = sio.loadmat(mat_fname)
            id_neuron = id_variable['input_id']
            self.activity_list = [id_neuron[0][i][0] for i in range(id_neuron.shape[1])]
            self.behavior_list = ['10^-4 2,3-pentanedione','10^-4 2-butanone','100mM NaCL']
            trace = trace_variable['trace_array'].T
            time = trace_variable['times']
            stimulate_seconds = trace_variable['stimulus_seconds']
            worm_trace = self.process_trace(trace,clip_fraction)
            self.activity_worms =  np.zeros([total_experiment,worm_trace.shape[0],worm_trace.shape[1]])
            self.behavior_worms = np.zeros([total_experiment,worm_trace.shape[0],stimulate_seconds.shape[0]])
            # neural activity target
            self.activity_worms[if_name,:,:] = worm_trace
            
            # generate the one-hot encoded presenetation for odor input
            for it_stimu in range(stimulate_seconds.shape[0]):
                tim1_ind = time>stimulate_seconds[it_stimu][1][0][0]
                tim2_ind = time<stimulate_seconds[it_stimu][2][0][0]
                self.behavior_worms[if_name,:,it_stimu] = np.squeeze(np.multiply(tim1_ind.astype(np.int),tim2_ind.astype(np.int)))[int(trace.shape[0]*clip_fraction):trace.shape[0]]
            
    def process_trace(self,trace,clip_fraction):
        clip_trace = trace[int(trace.shape[0]*clip_fraction):trace.shape[0],:]
        #normalization based on mean, variance
        worm_trace = (clip_trace - np.nanmean(clip_trace))/np.nanstd(clip_trace)
        return worm_trace

    def process_position(self,position_neuron,clip_fraction):
        clip_position = trace[int(position_neuron.shape[0]*clip_fraction):position_neuron.shape[0],:]
        #normalization based on mean, variance
        neuron_position = (clip_position - np.nanmean(clip_position))/np.nanstd(clip_position)
        return neuron_position

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
        upsample_factor = 0.2494/dt
        upsampled_worms = []
        for worm_id in worm_list:
            #(T,D) = self.behavior_matrix_list[worm_id].shape
            # currently not perform missing behavior detection
            (T,D) = self.behavior_worms[worm_id].shape
            interp_matrix = np.zeros(shape = (int(T*upsample_factor)+1, D))
            for i in range(D):
                inter_points = np.linspace(0,T, int(T*upsample_factor+1))
                interp_matrix[:,i] = np.interp(inter_points, range(T), self.behavior_worms[worm_id][:,i])
                
                #perform missing behavior detection
                #interp_matrix[:,i] = np.interp(inter_points, range(T), self.behavior_matrix_list[worm_id][:,i])
            interp_df = pd.DataFrame(interp_matrix, columns = self.behavior_list)
            #upsampled_worms.append(interp_df)
            upsampled_worms.append(interp_matrix)
        return upsampled_worms

    def interpolate_activity(self, worm_list,  dt):
        '''
        Parameters:
            worm_list - list of worm numbers
            dt - time step size
        
        Returns:
            list of Dataframes containing upsampled activity data for each worm 
        '''
        upsample_factor = 0.2494/dt
        upsampled_worms = []
        for worm_id in worm_list:
            (T,D) = self.activity_matrix_list[worm_id].shape
            interp_matrix = np.zeros(shape = (int(T*upsample_factor)+1, D))
            for i in range(D):
                inter_points = np.linspace(0,T, T*upsample_factor+1)
                interp_matrix[:,i] = np.interp(inter_points, range(T), self.activity_matrix_list[worm_id][:,i])
            interp_df = pd.DataFrame(interp_matrix, columns = self.activity_list[worm_id])
            #upsampled_worms.append(interp_df)
            upsampled_worms.append(interp_matrix)
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
        self.data = data  # (timesteps, n_neurons)
        self.window_size = window_size
        self.max_index = self.data[0].shape[0] // self.window_size[0] - 2
        self.crop_target = int((data_param_dict['k1'] - 1 + (data_param_dict['k2'] - 1)/data_param_dict['up1_factor'] + (data_param_dict['k4'] - 1)/(data_param_dict['up1_factor'] * data_param_dict['up2_factor']))/2)
        self.crop_stim_feature = int((data_param_dict['k3'] - 1   + data_param_dict['k4'] - 1)/2)
        

    def __getitem__(self, index):
        return self.data[0][(index+1)*self.window_size[0] - self.crop_stim_feature:(index+2)*self.window_size[0] + self.crop_stim_feature, :], self.data[1][(index+1)*self.window_size[1] - self.crop_target:(index+2)*self.window_size[1] + self.crop_target, :], self.data[2][(index+1)*self.window_size[2]:(index+2)*self.window_size[2], :], self.data[3][(index+1)*self.window_size[3]:(index+2)*self.window_size[3], :]

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
    
