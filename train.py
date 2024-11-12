import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from netCDF4 import Dataset,MFDataset
import matplotlib.pyplot as plt
import pandas as pd

torch.manual_seed(0) #makes training reproducible

class SSTDataset(torch.utils.data.Dataset):  #dataset class
    def __init__(self, data_point_csv, dataset,lon_window, lat_window,t_window, transform=None, target_transform=None):
        self.data = pd.read_csv(data_point_csv)

        self.dataset = dataset

        self.t_window = t_window
        self.lon_window = lon_window
        self.lat_window = lat_window
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        t = self.data.iloc[idx]['Time_Index']

        lat = self.data.iloc[idx]['Lat_Index']

        lon = self.data.iloc[idx]['Lon_Index']

        image = self.dataset[t:t+self.t_window,lat:lat+self.lat_window,lon:lon+self.lon_window]
        label = self.dataset[t+self.t_window+1,lat:lat+self.lat_window,lon:lon+self.lon_window]
        #TODO add normalization transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




files = ["1986_2000/1981_1985.nc","1986_2000/1986_1990.nc","1986_2000/1991_1995.nc","1986_2000/1996_2000.nc"]

#TODO add validation/test sets

t_window = 4
lon_window = 100
lat_window = 100
batch_size = 16
num_epochs = 1


train_dataset = MFDataset(files).variables['sst']

data_points = './data_points.csv'

dataset = SSTDataset(data_points,train_dataset,lon_window,lat_window,t_window)

train_dataLoader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

#TODO add masking so we arent sampling over land

#optimizer = torch.optim.AdamW(model.parameters(),lr=0.0002,betas=(.9, .95), eps=1e-8)
#model = model()

for epoch in range(num_epochs):
    for image,labels in iter(train_dataLoader): #iterates through dataset


        #run through model
        #TODO build model and update code below

        ''' commented out until the models are built
        
        pred = model(data)  #predict next timestep - pass data through entire model
        recon = EncoderDecoder(data) #try to reconstruct input 
        #could have the model output both pred and recon for better efficiency (dont have to pass inputs through the encoder twice)
        
        pred_loss = torch.nn.functional.mse_loss(pred,label)
        recon_loss = torch.nn.functional.mse_loss(recon,data)
        loss = l_pred*pred_loss + l_recon* recon_loss  #l_pred, and l_recon are hyperparameters weighting each loss term
        
        optimizer.zero_grad()
        
        loss.backward() #backprop
        optimizer.step() #update weights
        '''
        
        
        
        
        










