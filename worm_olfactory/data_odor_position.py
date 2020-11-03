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
            
            position_neuron_x = pos_variable['positions'][:,:,0].T
            for i_n in range(position_neuron_x.shape[1]):
                self.behavior_list.append('%s_x' %(id_neuron[0][i_n][0]))
            position_neuron_y = pos_variable['positions'][:,:,1].T
            for i_n in range(position_neuron_y.shape[1]):
                self.behavior_list.append('%s_y' %(id_neuron[0][i_n][0]))
            position_neuron_z = pos_variable['positions'][:,:,2].T
            for i_n in range(position_neuron_z.shape[1]):
                self.behavior_list.append('%s_z' %(id_neuron[0][i_n][0]))
            position_neuron = np.concatenate((position_neuron_x,position_neuron_y,position_neuron_z),axis = 1)
            
            time = trace_variable['times']
            stimulate_seconds = trace_variable['stimulus_seconds']
            worm_trace = self.process_trace(trace,clip_fraction)
            position_input = self.process_trace(position_neuron,clip_fraction)
            #pdb.set_trace()
            self.activity_worms =  np.zeros([total_experiment,worm_trace.shape[0],worm_trace.shape[1]])
            self.behavior_worms = np.zeros([total_experiment,worm_trace.shape[0],stimulate_seconds.shape[0] + position_input.shape[1]])
            # neural activity target
            self.activity_worms[if_name,:,:] = worm_trace
            
            # generate the one-hot encoded presenetation for odor input
            for it_stimu in range(stimulate_seconds.shape[0]):
                tim1_ind = time>stimulate_seconds[it_stimu][1][0][0]
                tim2_ind = time<stimulate_seconds[it_stimu][2][0][0]
                self.behavior_worms[if_name,:,it_stimu] = np.squeeze(np.multiply(tim1_ind.astype(np.int),tim2_ind.astype(np.int)))[int(trace.shape[0]*clip_fraction):trace.shape[0]]

            self.behavior_worms[if_name,:,3:self.behavior_worms.shape[2]] = position_input
            # current behavior term contains the odor input and position input.
            # currently not perform process to find missing behavior (all included)
            #self.behavior_matrix_list, self.missing_behavior = self.__process_behavior()
            #self.masked_behavior = [np.concatenate((self.behavior_matrix_list[i], self.missing_behavior[i]), axis = 0) for i in range(len(self.behavior_matrix_list))]
            
    def process_trace(self,trace,clip_fraction):
        clip_trace = trace[int(trace.shape[0]*clip_fraction):trace.shape[0],:]
        #normalization based on mean, variance
        #worm_trace = (clip_trace - np.nanmean(clip_trace))/np.nanstd(clip_trace)
        worm_trace = clip_trace
        # df/f0 is not performed, the detection of f0 is not intuitive
        # base_trace = nanmin(clip_trace)
        # worm_trace = (clip_trace - base_trace)/base_trace
        return worm_trace

    def process_position(self,position_neuron,clip_fraction):
        clip_position = trace[int(position_neuron.shape[0]*clip_fraction):position_neuron.shape[0],:]
        #normalization based on mean, variance
        neuron_position = (clip_position - np.nanmean(clip_position))/np.nanstd(clip_position)
        #neuron_position = clip_position
        return neuron_position

#    def order_neuron_df(self):
#        '''
#        Returns a dataframe containing all the neurons including neuron_subset such that all neurons
#        in the subset are at the end of the dataframe
#
#        Returns:
#            ordered dataframe of neurons
#
#        *The output of this function is always stored in self.neuron when the class is initialized
#        '''
#        outer_neurons = self.__neuronsdf.drop(index = self.activity_list)
#        mating_neurons = self.__neuronsdf.drop(index = list(outer_neurons.index))
#        reordered_neurons = pd.concat([outer_neurons, mating_neurons],axis = 0, sort = False)
#        return reordered_neurons
#
#    def extract_subnet(self):
#        '''
#        Read from file designating synapses exclusive to the neurons we are interested.
#
#        Returns:
#            pandas dataframe of relevant synapses
#        '''
#        pd.options.mode.chained_assignment = None
#        synapses = self.__netdf.dropna(axis=1, how = 'all')
#        sub_synapses = synapses[synapses['from'].isin(self.neuron_list)]
#        sub_synapses = sub_synapses[sub_synapses['to'].isin(self.neuron_list)]
#        sub_synapses = sub_synapses.reset_index(drop=True)
#        return sub_synapses
#
#    def extract_subneuron(self):
#        '''
#        Read from file specifying neuron transmitters and annotations that we are interested in .
#
#        Returns:
#            panda dataframe of relevant neurons
#        '''
#        pd.options.mode.chained_assignment = None
#        neurons_subset = self.neurons.dropna(axis=1, how='all')
#        neurons_subset.drop(neurons_subset.index.difference(self.activity_list),0,inplace = True)
#        return neurons_subset
#
#    def write_data_subset(self, net_file: str, neuron_file: str) -> None:
#        '''
#        Write extracted data to new files
#
#        Parameters:
#            net_file: new file location of net_subset
#        '''
#        self.extract_subnet(self.neuron_list).to_csv(net_file)
#        self.extract_subneuron(self.neuron_list).to_csv(neuron_file)
#
#    def get_calcium_data(self, dataset = None):
#        '''
#        Returns calcium imaging data of all relevant neurons
#
#        Parameters:
#            dataset = None: integer (0-6) that specifies which dataset to retrieve. defaults to return a list of all of them
#
#        Returns:
#            pandas dataframe of relevant data or list of dataframes
#        '''
#        if dataset != None:
#            activity_data = self.activity[dataset][self.neuron_list]
#        else:
#            activity_data = []
#            for data in self.activity:
#                activity_data.append(data[self.neuron_list])
#        return activity_data
#
#    def get_behavior_data(self, dataset = None):
#        '''
#        Returns behavior data of all relevant neurons
#
#        Parameters:
#                dataset = None: integer (0-6) that specifies which dataset to retrieve. defaults to return a list of all of them
#
#        Returns:
#                pandas dataframe of relevant data or list of dataframes
#        '''
#        if dataset != None:
#                behavior_data = self.activity[dataset][self.behavior_list]
#        else:
#                behavior_data = []
#                for data in self.activity:
#                        activity_data.append(data[self.behavior_list])
#        return behavior_data

    def __process_activity(self):
        '''
        Returns a list of matrices corresponding to the data missing in the activity columns of the activity_worms dataframes and
        a matrix of the activity with NaNs replaced by 0's
        '''
        missing_data, activity_data = [],[]
        for worm in self.activity_worms:
            worm = (worm - worm.mean())/worm.std()
            #TODO: will future perform df/f normalization to each neuron
            act_matrix = worm.values
            missing_act = np.zeros(act_matrix.shape)
            missing_act[np.isnan(act_matrix)] = 1
            act_matrix[np.isnan(act_matrix)] = 0
            missing_data.append(missing_act)
            activity_data.append(act_matrix)
        return activity_data, missing_data

    def __process_behavior(self):
        missing_behavior, behavior_matrices = [], []
        for worm in self.behavior_worms:
            worm.loc[worm['LtoVul']==1,['LtoVul']] = np.NaN
            worm.loc[worm['LtoTips']==1,['LtoTips']] = np.NaN
            worm.loc[:,worm.columns != 'PostSperm'] = (worm.loc[:, worm.columns != 'PostSperm'] - worm.loc[:,worm.columns!= 'PostSperm'].mean())/worm.loc[:, worm.columns != 'PostSperm'].std()
            behavior_matrix = worm.values
            missing_beh = np.zeros(behavior_matrix.shape)
            missing_beh[np.isnan(behavior_matrix)] = 1
            behavior_matrix[np.isnan(behavior_matrix)] = 0
            behavior_matrices.append(behavior_matrix)
            missing_behavior.append(missing_beh)
        return behavior_matrices, missing_behavior  
    
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
                inter_points = np.linspace(0,T, T*upsample_factor+1)
                interp_matrix[:,i] = np.interp(inter_points, range(T), self.behavior_worms[worm_id][:,i])
                #interp_matrix[:,i] = np.interp(inter_points, range(T), self.behavior_matrix_list[worm_id][:,i])
                # currently not perform missing behavior detection
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

def generate_stimulus_features(input_df_list, num_bins = 10):
    '''
    Create a list of stimulus features for each worm. Each element of the list is the data associated with a different worm
    'LtoTips', 'LtoVul', 'pos_x', 'pos_y' - split into several features that are their values filtered through a set of Gaussians with different means 
    'Rel. Velocity', 'Velocity' - split into two features, one for the positive and one for the negative values. both features will be positive

    If you pass input_df_list = None, the function will return the stimulus features applied to the original data for all worms in a list

    Parameters:  
        input_df: list of dataframes that we want to generate features of 
        num_bins = 10: how many gaussians we filter the distance associated values into

    Returns: (LtoTips, LtoVul, pos_x, pos_y, Velocity, Rel.Velocity)
    '''

    behavior_input_list = []
    std_scale = 1
    eps = 1e-6
    for d in input_df_list:
        # define variables necessary to do feature extraction for distance variables
        LtoTips_min, LtoTips_max = d['LtoTips'].min(), d['LtoTips'].max()
        LtoVul_min, LtoVul_max = d['LtoVul'].min(), d['LtoTips'].max()
        posx_min, posx_max = d['pos_x'].min(), d['pos_x'].max()
        posy_min, posy_max = d['pos_y'].min(), d['pos_y'].max()
        x_min, x_max = d['x'].min(), d['x'].max()
        y_min, y_max = d['y'].min(), d['y'].max()
        LtoTips_means = np.linspace(LtoTips_min, LtoTips_max, num = num_bins, endpoint = False)
        LtoVul_means = np.linspace(LtoVul_min, LtoVul_max, num = num_bins, endpoint = False)
        posx_means = np.linspace(posx_min, posx_max, num = num_bins, endpoint = False)
        posy_means = np.linspace(posy_min, posy_max, num = num_bins, endpoint = False)
        x_means = np.linspace(x_min, x_max, num = num_bins, endpoint = False)
        y_means = np.linspace(y_min, y_max, num = num_bins, endpoint = False)
        LtoTips_std = ( LtoTips_max - LtoTips_min)/(std_scale*num_bins)
        LtoVul_std = ( LtoVul_max - LtoVul_min)/(std_scale*num_bins)
        posx_std = ( posx_max - posx_min)/(std_scale*num_bins)
        posy_std = ( posy_max - posy_min)/(std_scale*num_bins)
        x_std = (x_max - x_min)/(std_scale*num_bins)
        y_std = (y_max - y_min)/(std_scale*num_bins)
        LtoTips_features =  np.zeros(shape = (d['LtoTips'].values.shape[0], num_bins))
        LtoVul_features = np.zeros(shape = (d['LtoVul'].values.shape[0], num_bins))
        posx_features = np.zeros(shape = (d['pos_x'].values.shape[0], num_bins))
        posy_features = np.zeros(shape = (d['pos_y'].values.shape[0], num_bins))
        x_features = np.zeros(shape = (d['x'].values.shape[0], num_bins))
        y_features = np.zeros(shape = (d['y'].values.shape[0], num_bins))
        for i in range(d['LtoTips'].values.shape[0]):
            for j in range(num_bins):
                LtoTips_features[i,j] = np.exp(-(d['LtoTips'].values[i] - LtoTips_means[j])**2/(LtoTips_std+eps)**2)
                LtoVul_features[i,j] = np.exp(-(d['LtoVul'].values[i] - LtoVul_means[j])**2/(LtoVul_std+eps)**2)
                posx_features[i,j] = np.exp(-(d['pos_x'].values[i] - posx_means[j])**2/(posx_std+eps)**2)
                posy_features[i,j] = np.exp(-(d['pos_y'].values[i] - posy_means[j])**2/(posy_std+eps)**2)
                x_features[i,j] = np.exp(-(d['x'].values[i] - x_means[j])**2/(x_std+eps)**2)
                y_features[i,j] = np.exp(-(d['y'].values[i] - y_means[j])**2/(y_std+eps)**2)

        # define variables necessary to split velocity variables into positive and negative
        relvelocity_pos, relvelocity_neg  = np.zeros(shape = (d['Rel.Velocity'].values.shape[0],1)), np.zeros(shape = (d['Rel.Velocity'].values.shape[0],1))
        velocity_pos, velocity_neg = np.zeros(shape = (d['Velocity'].values.shape[0],1)), np.zeros(shape = (d['Velocity'].values.shape[0],1))
        for i in range(d['Velocity'].values.shape[0]):
            if d['Velocity'].values[i] >= 0:
                velocity_pos[i] = d['Velocity'].values[i] 
            else:
                velocity_neg[i] = - d['Velocity'].values[i]
            if d['Rel.Velocity'].values[i] >= 0:
                relvelocity_pos[i] = d['Rel.Velocity'].values[i]
            else:
                relvelocity_neg[i] = -d['Rel.Velocity'].values[i]
        behavior_matrix = np.concatenate((LtoTips_features, 
            LtoVul_features, 
            d[['PostSperm','Rel.Speed']].values, 
            relvelocity_pos, relvelocity_neg, 
            d[['Speed','Spicules', 'TailAngle']].values,
            velocity_pos, velocity_neg, x_features, y_features,
            posx_features, posy_features), axis = 1)
        behavior_input_list.append(behavior_matrix)
    return behavior_input_list

class TimeSeriesDataloader(Dataset):
    def __init__(self, data, window_size): # trace.shape = (tpoints)
        super(TimeSeriesDataloader, self).__init__()
        self.data = data  # (timesteps, n_neurons)
        self.window_size = window_size
        self.max_index = self.data[0].shape[0] // self.window_size[0]

    def __getitem__(self, index):
        return self.data[0][index*self.window_size[0]:(index+1)*self.window_size[0], :], self.data[1][index*self.window_size[1]:(index+1)*self.window_size[1], :], self.data[2][index*self.window_size[2]:(index+1)*self.window_size[2], :]

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
    
