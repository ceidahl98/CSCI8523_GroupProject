import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
from netCDF4 import Dataset,MFDataset
import matplotlib.pyplot as plt

files = ["1986_2000/1981_1985.nc","1986_2000/1986_1990.nc","1986_2000/1991_1995.nc","1986_2000/1996_2000.nc"]

#TODO add validation/test sets

train_dataset = MFDataset(files)

x = np.arange(0,720)
plt.imshow(train_dataset.variables["sst"][0,x[::-1],:],)
plt.show()

lat = torch.arange(0,train_dataset.variables["sst"].shape[1])

lon = torch.arange(0,train_dataset.variables["sst"].shape[2])
time = torch.arange(0,train_dataset.variables["sst"].shape[0])


#TODO add masking so we arent sampling over land




def getBatch(dataset,batch_size,lat_window,lon_window,time,lat,lon):
    time = torch.randperm(len(time))[:batch_size]
    lon = torch.randperm(len(lon)-lon_window)[:batch_size]

    lat = torch.randperm(len(lat)-lat_window)[:batch_size]

    data = []
    labels = []
    #print(dataset.variables["sst"][time[0],])
    for i in range(batch_size):

        data.append(torch.tensor(dataset.variables["sst"][time[i],lat[i]:lat[i]+lat_window,lon[i]:lon[i]+lon_window]))
        labels.append(torch.tensor(dataset.variables["sst"][time[i]+1,lat[i]:lat[i]+lat_window,lon[i]:lon[i]+lon_window]))
        if data[i].shape[0]==0:
            print(data[i])

    data = torch.stack(data,dim=0)
    labels = torch.stack(labels,dim=0)
    #TODO normalize image

    return data, labels

lon_window = 150
lat_window = 100
batch_size = 16
num_epochs = 1
samples_per_epoch = 1e6
num_batches = int(samples_per_epoch/batch_size)

#optimizer = torch.optim.AdamW(model.parameters(),lr=0.0002,betas=(.9, .95), eps=1e-8)
#model = model()

for epoch in range(num_epochs):
    for _ in range(num_batches):
        data, labels = getBatch(train_dataset,batch_size,lat_window,lon_window,time,lat,lon)
        print(data.shape)

        #run through model
        #TODO build model and update code below

        ''' commented out until the models are built
        
        pred = model(data)  #predict next timestep - pass data through entire model
        recon = EncoderDecoder(data) #try to reconstruct input - enforce encoder and decoder are inverse of eachother
        #could have the model output both pred and recon for better efficiency (dont have to pass inputs through the encoder twice)
        
        pred_loss = torch.nn.functional.mse_loss(pred,label)
        recon_loss = torch.nn.functional.mse_loss(recon,data)
        loss = l_pred*pred_loss + l_recon* recon_loss  #l_pred, and l_recon are hyperparameters weighting each loss term
        
        optimizer.zero_grad()
        
        loss.backward() #backprop
        optimizer.step() #update weights
        
        
        
        
        
        
        '''









