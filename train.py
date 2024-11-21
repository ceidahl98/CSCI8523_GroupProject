import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from netCDF4 import Dataset,MFDataset
import matplotlib.pyplot as plt
import pandas as pd
from EncoderDecoder import Encoder, Decoder

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




files = ["data/1981_1985.nc", "data/1986_1990.nc",
             "data/1991_1995.nc", "data/1996_2000.nc",
             "data/2001_2005.nc", "data/2006_2010.nc",
             "data/2011_2015.nc", "data/2016_2020.nc",
             "data/2021_2024.nc"]
# files = ["data/1981_1985.nc", "data/1986_1990.nc"]
#TODO add validation/test sets


t_window = 4
lon_window = 16
lat_window = 16
batch_size = 16
num_epochs = 1


train_dataset = MFDataset(files).variables['sst']

data_points = './data_points.csv'

dataset = SSTDataset(data_points,train_dataset,lon_window,lat_window,t_window)

train_dataLoader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

#TODO add masking so we arent sampling over land

#optimizer = torch.optim.AdamW(model.parameters(),lr=0.0002,betas=(.9, .95), eps=1e-8)
#model = model()
encoder = Encoder()
decoder = Decoder()

encoder_optim = torch.optim.AdamW(encoder.parameters(),lr=0.0002,betas=(.9, .95), eps=1e-8)
decoder_optim = torch.optim.AdamW(decoder.parameters(),lr=0.0002,betas=(.9, .95), eps=1e-8)

loss = nn.MSELoss()
count = 0
for epoch in range(num_epochs):
    for image,labels in iter(train_dataLoader): #iterates through dataset
        count+=1
        image[image==image.min()]=-100 # I'm sure there is a better way to do this up above, I (Jack) will look into it
        #run through model
        #TODO build model and update code below
        hidden = encoder(image)
        recon = decoder(hidden) #try to reconstruct input 
        recon_loss = loss(recon,image)

        # Get input image and reconstruction
        # if count%1000==0:
        #     filename = f"comparison at {count}"
        #     plt.figure()
        #     plt.subplot(121)
        #     plt.imshow(image[0][0].detach().numpy())
        #     plt.title("Image")
        #     plt.subplot(122)
        #     plt.imshow(recon[0][0].detach().numpy())
        #     plt.title("Reconstruction")
        #     plt.text(0,0,f"recon loss: {recon_loss}")
        #     plt.savefig(filename)
        # print(recon_loss)
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(image[0][0].detach().numpy())
        # plt.subplot(122)
        # plt.imshow(recon[0][0].detach().numpy())
        # plt.show()
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        recon_loss.backward()
        encoder_optim.step()
        decoder_optim.step()
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

        
        
        
        










