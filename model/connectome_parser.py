import numpy as np
import pandas as pd
from pathlib import Path
import torch
import sys

__all__ = ['Connectome','Neuron']

class Neuron:
    def __init__(self, name, index, sex_spec, shared, sensory, motor):
        self.name = name
        self.index = index
        self.sex_spec = sex_spec
        self.shared = shared
        self.sensory = sensory
        self.motor = motor

class ConnectomeData:
    def __init__(self, cnctm_path, device = 'cuda'):
        '''
            Connectome class for Cook et al. (Nature 2019) on C.elegans connectome. Converts csv files into a set of dictionaries storing
            neuron and synaptic information
            basepath  = '../../cnctm_data' || '../../herm_cnctm_data'
            Attributes:
            neuron_list: list of all neurons objects in order of adjacency matrices.
            neuron_mask_dict: dictionary of masks that specify subsets of various neuron types (male, sex shared, sensory, motor)
            name_neuron_dict: dictionary that maps neuron names to neuron objects
            synapses_dict: dictionary that stores adjacency and weight matrices of chemical and electrical synapses
            '''
        basepath = cnctm_path
        self.__device = device
        self.__chemdf = pd.read_csv(basepath+'chem.csv', skipinitialspace = True, quotechar = '"').drop('Unnamed: 0', axis = 1)
        self.__gjsym = pd.read_csv(basepath+'elec_symmetric.csv', skipinitialspace = True, quotechar = '"').drop('Unnamed: 0', axis = 1)
        self.__gjassym = pd.read_csv(basepath+'elec_asymmetric.csv', skipinitialspace = True, quotechar = '"').drop('Unnamed: 0', axis = 1)
        if 'herm_cnctm_data' in basepath:
            self.__spec_neurondf = pd.read_csv(basepath + 'herm_specific_cells.csv', skipinitialspace = True, quotechar = '"')
            self.__spec_neurondf.index = list(self.__spec_neurondf['name'])
            self.__spec_neurondf.drop('name', axis = 1)
        else:
            self.__validated_symmetrized = pd.read_csv(basepath + 'connectome_validated_symmetrized.csv', skipinitialspace = True, quotechar = '"')
            self.__spec_neurondf = pd.read_csv(basepath + 'male_specific_cells.csv', skipinitialspace = True, quotechar = '"')
            self.__spec_neurondf.index = list(self.__spec_neurondf['name'])
            self.__spec_neurondf.drop('name', axis = 1)
        self.__chemdf.index = list(self.__chemdf.columns)
        self.__gjsym.index = list(self.__gjsym.columns)
        self.__gjassym.index = list(self.__gjassym.columns)
        self.__shared_neurondf = pd.read_csv(basepath + 'sex_shared_cells.csv', skipinitialspace = True, quotechar = '"')
        self.__shared_neurondf.index = list(self.__shared_neurondf['name'])
        self.__shared_neurondf.drop('name', axis = 1)

        self.neuron_list = self.__generate_neuron_list()
        self.neuron_mask_dict = self.__generate_neuron_masks()
        self.name_neuron_dict = {neuron.name:neuron for neuron in self.neuron_list}
        self.synapses_dict = self.__generate_weight_matrices()
        self.N = len(self.neuron_list)
        if 'herm_cnctm_data' not in basepath:
            self.update_synaptic_weights()

    def __generate_neuron_list(self):
        neuron_list = []
        for i, neuron_name in enumerate(list(self.__chemdf.columns)):
            if neuron_name in list(self.__shared_neurondf['name']):
                neuron_list.append(Neuron(name = neuron_name,
                                          index = i,
                                          sex_spec = False,
                                          shared = True,
                                          sensory = (self.__shared_neurondf.loc[neuron_name, 'cell type'] == 'sensory'),
                                          motor = (self.__shared_neurondf.loc[neuron_name, 'cell type'] == 'motorneuron')
                                          ))
            if neuron_name in list(self.__spec_neurondf['name']):
                neuron_list.append(Neuron(name = neuron_name,
                                          index = i,
                                          sex_spec = True,
                                          shared = False,
                                          sensory = (self.__spec_neurondf.loc[neuron_name, 'cell type'] == 'sensory'),
                                          motor = (self.__spec_neurondf.loc[neuron_name, 'cell type'] == 'motorneuron')
                                          ))
        return neuron_list

    def __generate_neuron_masks(self):
        sex_spec_mask = torch.zeros(size = (1, len(self.neuron_list)), device = self.__device)
        shared_mask = torch.zeros(size = (1, len(self.neuron_list)), device = self.__device)
        motor_mask = torch.zeros(size = (1, len(self.neuron_list)), device = self.__device)
        sensory_mask = torch.zeros(size = (1, len(self.neuron_list)), device = self.__device)
        for i, neuron in enumerate(self.neuron_list):
            if neuron.sex_spec:
                sex_spec_mask[:,i] = 1
            if neuron.shared:
                shared_mask[:,i] = 1
            if neuron.motor:
                motor_mask[:,i] = 1
            if neuron.sensory:
                sensory_mask[:,i]= 1
        return {'sex_spec': sex_spec_mask, 'shared': shared_mask, 'motor': motor_mask, 'sensory': sensory_mask}

    def __generate_weight_matrices(self):
        chem_weights = torch.from_numpy(self.__chemdf.values).transpose(0,1).float().to(self.__device)
        esym_weights = torch.from_numpy(self.__gjsym.values).transpose(0,1).float().to(self.__device)
        eassym_weights = torch.from_numpy(self.__gjassym.values).transpose(0,1).float().to(self.__device)
        chem_weights[torch.isnan(chem_weights)] = 0
        esym_weights[torch.isnan(esym_weights)] = 0
        eassym_weights[torch.isnan(eassym_weights)] = 0
        chem_adj = torch.where(chem_weights == 0, torch.tensor([0.], device = self.__device), torch.tensor([1.], device = self.__device))
        esym_adj = torch.where(esym_weights == 0, torch.tensor([0.], device = self.__device), torch.tensor([1.], device = self.__device))
        eassym_adj = torch.where(eassym_weights == 0, torch.tensor([0.], device = self.__device), torch.tensor([1.], device = self.__device))
        return {'chem_weights': chem_weights, 'esym_weights': esym_weights, 'eassym_weights': eassym_weights,
                'chem_adj': chem_adj, 'esym_adj': esym_adj, 'eassym_adj': eassym_adj}

    def make_full_connectome_activity(self, activity_data, activity_neuron_list, ablation_list = []):
        '''
        Takes partial neural activity data (B, N_s, T) and maps it into a tensor (B, N, T) where N includes all neurons in this connectome.
        All missing values are NaN. For this connectome N = 302. 

        Parameters:
            activity_data: tensor of calcium traces from a single worm (B, N_s, T)
            activity_neuron_list: list of neuron labels associated with calcium traces

        Return:
            all_activity: tensor (B, N, T)
        '''   
        all_activity = np.NaN*torch.ones(size = (activity_data.shape[0], len(self.neuron_list), activity_data.shape[2]), device = self.__device)
        for i in range(activity_data.shape[1]):
            if activity_neuron_list[i] not in ablation_list:
                mask_neuron_index = self.name_neuron_dict[activity_neuron_list[i]].index # map neural activity to full connectome tensor index
                all_activity[:,mask_neuron_index,:] = activity_data[:,i,:] # fill in neural activity to full neuron (B, N, T) activity vector
        return all_activity
	
    def update_synaptic_weights(self):
        presyn_neurons = self.__validated_symmetrized['from'].values
        postsyn_neurons = self.__validated_symmetrized['to'].values
        synapse_types = self.__validated_symmetrized['synaps'].values
        weights = self.__validated_symmetrized['number'].values
        for k in range(presyn_neurons.shape[0]):
            if presyn_neurons[k] and postsyn_neurons[k] in self.name_neuron_dict:
                if presyn_neurons[k] == presyn_neurons[k] and postsyn_neurons[k] == postsyn_neurons[k] and weights[k] == weights[k]:
                    i = self.name_neuron_dict[postsyn_neurons[k]].index
                    j = self.name_neuron_dict[presyn_neurons[k]].index
                    if synapse_types[k] == 'electrical':
                        self.synapses_dict['esym_weights'][i,j] = weights[k]
                    elif synapse_types[k] == 'chemical':
                        self.synapses_dict['chem_weights'][i,j] = weights[k]

        self.synapses_dict['eassym_weights'] = self.synapses_dict['esym_weights']
        for i in range(len(self.neuron_list)):
            for j in range(len(self.neuron_list)):
                if i > j:
                    self.synapses_dict['eassym_weights'][i,j] = 0 
        self.synapses_dict['chem_adj'] = torch.where(self.synapses_dict['chem_weights'] == 0, torch.tensor([0.], device = self.__device), torch.tensor([1.], device = self.__device))
        self.synapses_dict['esym_adj'] = torch.where(self.synapses_dict['esym_weights'] == 0, torch.tensor([0.], device = self.__device), torch.tensor([1.], device = self.__device))
        self.synapses_dict['eassym_adj'] = torch.where(self.synapses_dict['eassym_weights'] == 0, torch.tensor([0.], device = self.__device), torch.tensor([1.], device = self.__device))

    def symmetrize_activity(self, target_data):
        for key in self.name_neuron_dict:
            if (key[-1] == 'L') and (key[:-1] != 'R1A' or key[:-1] != 'PVP'):
                right_neuron = key[:-1] + 'R'
                if right_neuron in self.name_neuron_dict:
                    right_neuron_index = self.name_neuron_dict[key[:-1] + 'R'].index
                    left_neuron_index = self.name_neuron_dict[key].index
                    target_data[:,right_neuron_index] = target_data[:,left_neuron_index]
        return target_data


if __name__ == "__main__":
    c = ConnectomeData('../herm_cnctm_data/')
