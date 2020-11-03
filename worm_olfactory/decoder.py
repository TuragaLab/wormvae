import torch
import torch.nn as nn
import numpy as np


class Decoder(torch.nn.Module):
    def __init__ (self, n_feature, n_output):
        """
        Single Layer Linear Neural Network
        """
        super(Decoder, self).__init__()
        self.linear = torch.nn.Linear(n_feature, n_output)
        self.n_feature = n_feature
        self.n_output = n_output
        
    def weighted_MSELoss(self,pred, target, missing):
        loss = 0.5*torch.mean(-(missing- 1)*(pred - target)**2)
        return loss
    
    def __MSELoss(self, pred, target):
        loss = 0.5*torch.mean((pred - target)**2)
        return loss
    
    def forward(self, x):
        return self.linear(x)      
    
    def predict(self, in_data, missing_in, missing_out, target_data):
        """
        Predict using trained network assuming missing output data.
        
        Parameters:
        in_data, out_dim
        missing_in - matrix of missing input_data values
        
        Returns:
        
        prediction values in tensor form
        """
        out_data = torch.zeros(in_data.shape[0], self.n_output)
        input_array = np.concatenate((in_data,missing_in),1) 
        miss_out_tensor = torch.from_numpy(missing_out).unsqueeze(0).to(torch.float32).cuda()
        target_tensor = torch.from_numpy(target_data).unsqueeze(0).to(torch.float32).cuda()
        loss_sum = 0
        for image in range(in_data.shape[0]):
            in_tensor = torch.from_numpy(input_array[image,:]).unsqueeze(0).to(torch.float32).cuda()
            output = self.forward(in_tensor)
            out_data[image,:] = output
            loss_sum += self.weighted_MSELoss(output, target_tensor[:,image,:], miss_out_tensor[:,image,:])
        loss = loss_sum/(in_data.shape[0])
        return out_data, loss

class Scale_Net(torch.nn.Module):
    def __init__(self, n_feature, stdev, mean):
            super(Scale_Net, self).__init__()
            self.n_feature = n_feature
            self.scale = torch.from_numpy(stdev).unsqueeze(0).to(torch.float32)
            self.bias = torch.from_numpy(mean).unsqueeze(0).to(torch.float32)
            self.scale = torch.nn.Parameter(self.scale)
            self.bias = torch.nn.Parameter(self.bias)
            
    def forward(self,x):
            return (x - self.bias)/self.scale
        
    def predict(self, in_data):
            out_data = torch.zeros(in_data.shape[0], self.n_feature)
            for image in range(in_data.shape[0]):
                in_tensor = torch.from_numpy(in_data[image,:]).unsqueeze(0).to(torch.float32).cuda()
                output = self.forward(in_tensor)
                out_data[image,:] = output
            return out_data

def train(decoder, train_act_list, missing_act_list, train_bev_list, missing_bev_list, learning_rate, epochs, batch_size):
        loss_list, in_tensor_list, target_list, missing_in_list, missing_out_list = [], [], [], [], []
        criterion = decoder.weighted_MSELoss
        optimizer = torch.optim.Adam(decoder.parameters(), lr = learning_rate)
        for j in range(len(train_act_list)):
            in_tensor_list.append(torch.from_numpy(train_act_list[j])).unsqueeze(0).to(torch.float32).cuda()
            target_list.append(torch.from_numpy(train_bev_list[j]).unsqueeze(0).to(torch.float32).cuda())
            missing_in_list.append(torch.from_numpy(missing_act_list[j]).unsqueeze(0).to(torch.float32).cuda())
            missing_out_list.append(torch.from_numpy(missing_bev_list[j]).unsqueeze(0).to(torch.float32).cuda())        
        for epoch in range(epochs):
            loss_sum = 0
            batch_num = int(in_tensor_list[].shape[1]/batch_size)
            for i in range(batch_num):
                optimizer.zero_grad()
                if i < batch_num - 1:
                    in_tuple = tuple([in_tensor_list[j][:,i*batch_size:(i+1)*batch_size,:] for j in range(len(train_act_list))])
                    miss_tuple = tuple([missing_in_list[j][:,i*batch_size:(i+1)*batch_size,:] for j in range(len(train_act_list))])
                    act_batch, missing_batch = torch.cat((in_tuple),dim = 1), torch.cat((miss_tuple), dim = 1)
                    in_batch = torch.cat((act_batch, miss_batch), dim = 0)
                    cat_input = torch.cat((neuron_in_tensor[:,i*batch_size:(i+1)*batch_size,:], missing_in_tensor[:, i*batch_size:(i+1)*batch_size,:]),2)
                    output = decoder.forward(cat_input)
                    loss = criterion(output,target[:,i*batch_size:(i+1)*batch_size,:],missing_out_tensor[:,i*batch_size:(i+1)*batch_size,:])
                else:
                    cat_input = torch.cat((neuron_in_tensor[:,i*batch_size:,:], missing_in_tensor[:, i*batch_size:,:]),2)
                    output = decoder.forward(cat_input)
                    loss = criterion(output,target[:,i*batch_size:,:],missing_out_tensor[:,i*batch_size:,:])
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            epoch_loss = loss_sum/(len(train_act_list)*batch_num)
            loss_list.append(epoch_loss)
        return loss_list

if __name__ == "__main__":
