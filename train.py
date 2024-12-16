import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from netCDF4 import Dataset,MFDataset
import matplotlib.pyplot as plt
import pandas as pd
from EncoderDecoder import autoEncoder
from transformer import GPT, GPTConfig
import csv
#max = 38.58
#min = -1.8
#mean = .38
#std = .332
torch.manual_seed(42) #makes training reproducible
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

def save_model(auto_encoder, transformer_model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'auto_encoder_state_dict': auto_encoder.state_dict(),
        'transformer_model_state_dict': transformer_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved to {path}")

def normalize(x):
    max = 38.58
    min = -1.8

    return (x - min) / (max - min)

def normalize_with_mask(tensor, mask):
    tensor_min = tensor[mask].min()
    tensor_max = tensor[mask].max()
    tensor[mask] = (tensor[mask] - tensor_min) / (tensor_max - tensor_min)
    tensor[~mask] = .5
    return tensor

class SSTDataset(torch.utils.data.Dataset):  #dataset class
    def __init__(self, data_point_csv, dataset,lon_window, lat_window,t_window, transform=None, target_transform=None, offset=0):
        self.data = data_point_csv

        self.dataset = dataset

        self.t_window = t_window
        self.lon_window = lon_window
        self.lat_window = lat_window
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        t = self.data.iloc[idx]['Time_Index']

        lat = self.data.iloc[idx]['Lat_Index']

        lon = self.data.iloc[idx]['Lon_Index']
        print(self.dataset.shape)
        image = self.dataset[t:t+self.t_window,lat:lat+self.lat_window,lon:lon+self.lon_window]
        label = self.dataset[t+self.t_window+1,lat:lat+self.lat_window,lon:lon+self.lon_window]

        ocean_mask = image > -100
        labels_mask = label > -100
        '''image[image < -100] = np.mean(image[ocean_mask])

        label[label < -100] = np.mean(label[labels_mask])'''

        image = normalize_with_mask(image,ocean_mask)
        label = normalize_with_mask(label,labels_mask)
        print(self.data.iloc[idx],"TEST")
        lats = torch.tensor(lat).repeat(4)

        lons = torch.tensor(lon).repeat(4)
        times = torch.tensor(t).repeat(4)
        if self.transform:
            image = self.transform(image).permute(1,0,2).unsqueeze(0)
        if self.target_transform:
            label = self.target_transform(label).unsqueeze(0)
        return image, label, lats,lons,times

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(.38,.332)
    ])




files = [r".\data\1981_1985.nc", r".\data\1986_1990.nc",
             r".\data\1991_1995.nc", r".\data\1996_2000.nc",
             r".\data\2001-2005.nc", r".\data\2006-2010.nc",
             r".\data\2011-2015.nc", r".\data\2016-2020.nc",
             r".\data\2021-2024.nc"]
# files = [r"D:\Datasets\data\1981_1985.nc", r"D:\Datasets\data\1986_1990.nc"]
#TODO add validation/test sets

if torch.cuda.is_available():
    DEVICE=torch.device('cuda:0')

else:
    DEVICE=torch.device("cpu")
print(DEVICE)

t_window = 4
lon_window = 16
lat_window = 16
batch_size = 32
num_epochs = 1

dataset = MFDataset(files).variables['sst']
historical_averages = np.load("historical_averages.npy")         #Reads in historical averages using data from 1986-2023, dimension is 366x720x1440

data_points = './data_points.csv'
csv_ = pd.read_csv(data_points)
total_len = int(csv_.shape[0])
train_len = int(total_len*38/40)

val_len = int(total_len*1/40)
test_len = total_len-train_len-val_len
train_csv = csv_.iloc[0:train_len,:]
val_csv = csv_.iloc[train_len+1:train_len+val_len,:]
test_csv = csv_.iloc[:-test_len,:]

train_dataset = SSTDataset(train_csv,dataset,lon_window,lat_window,t_window,transform=transform,target_transform=transform)
val_dataset = SSTDataset(val_csv,dataset,lon_window,lat_window,t_window,transform=transform,target_transform=transform)
test_dataset = SSTDataset(test_csv,dataset,lon_window,lat_window,t_window,transform=transform,target_transform=transform)
train_dataLoader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True)
val_dataLoader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)
test_dataLoader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)

#TODO add masking so we arent sampling over land

#optimizer = torch.optim.AdamW(model.parameters(),lr=0.0002,betas=(.9, .95), eps=1e-8)
#model = model()
in_channels = 1
embedding_dim=512

auto_encoder = autoEncoder(in_channels,embedding_dim).to(device)
transformer_model = GPT(GPTConfig,2048).to(device)
optim = torch.optim.AdamW(list(auto_encoder.parameters())+list(transformer_model.parameters()),lr=0.0002,betas=(.9, .95), eps=1e-8)

l_pred=1
l_recon=1
model_save_iter = 0
loss_fn = nn.MSELoss()
count = 0
checkpoint_path = './models/checkpoint.pt'
epoch_save_path = './models/'
csv_file_path = './models/loss_csv.csv'
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Epoch", "Val_Loss", "Test_Loss"])


    for epoch in range(num_epochs):
        for image,labels, lats,lons,times in iter(train_dataLoader): #iterates through dataset
            optim.zero_grad()
            print(lats.shape)
            count+=1


            #plt.imshow(image[0,0,0,:,:], cmap='viridis', aspect='auto', vmin=0, vmax=1)

            #plt.show()
            #run through model
            #TODO build model and update code below

            hidden = auto_encoder.encode(image) #(B,channels,time,features)
            recon = auto_encoder.decode(hidden)
            B, C, D, H, W = hidden.shape

            hidden = hidden.permute(0,2,3,4,1).flatten(start_dim=2)

            hidden = transformer_model(hidden,lats,lons)[:,-1,:].unsqueeze(1).view(B,C,1,H,W) #predict next state (B,C,1,features)


            pred = auto_encoder.decode(hidden)
            pred_loss = loss_fn(pred, labels)


            recon_loss = loss_fn(recon,image)
            loss = l_pred * pred_loss + l_recon * recon_loss
            #loss = recon_loss


            loss.backward()
            optim.step()
            model_save_iter+=1
            if model_save_iter % 10 ==0:
                save_model(auto_encoder, transformer_model, optim, epoch, loss, checkpoint_path)
            break

        with torch.no_grad():
            val_loss =0
            for image, labels, lats, lons, times in iter(val_dataLoader):

                hidden = auto_encoder.encode(image)  # (B,channels,time,features)
                recon = auto_encoder.decode(hidden)
                B, C, D, H, W = hidden.shape

                hidden = hidden.permute(0, 2, 3, 4, 1).flatten(start_dim=2)

                hidden = transformer_model(hidden, lats, lons)[:, -1, :].unsqueeze(1).view(B, C, 1, H,
                                                                                           W)  # predict next state (B,C,1,features)

                pred = auto_encoder.decode(hidden)
                pred_loss = loss_fn(pred, labels)

                recon_loss = loss_fn(recon, image)
                loss += l_pred * pred_loss + l_recon * recon_loss


                break
            test_loss=0
            for image, labels, lats, lons, times in iter(test_dataLoader):
                hidden = auto_encoder.encode(image)  # (B,channels,time,features)
                recon = auto_encoder.decode(hidden)
                B, C, D, H, W = hidden.shape

                hidden = hidden.permute(0, 2, 3, 4, 1).flatten(start_dim=2)

                hidden = transformer_model(hidden, lats, lons)[:, -1, :].unsqueeze(1).view(B, C, 1, H,
                                                                                           W)  # predict next state (B,C,1,features)

                pred = auto_encoder.decode(hidden)
                pred_loss = loss_fn(pred, labels)

                recon_loss = loss_fn(recon, image)
                loss += l_pred * pred_loss + l_recon * recon_loss

                break

            if epoch % 5 ==0:
                test_loss= test_loss/test_len
                val_loss = val_loss/val_len
                save_model(auto_encoder, transformer_model, optim, epoch, loss, epoch_save_path+'Epoch_'+str(epoch)+'.pt')
                writer.writerow((epoch,val_loss,test_loss))

            #for image,labels in iter(val_dataset):
        '''with torch.no_grad():
            recon = auto_encoder(test)
            hidden = auto_encoder.encode(test)
            B, C, D, H, W = hidden.shape
    
            hidden = hidden.permute(0, 2, 3, 4, 1).flatten(start_dim=2)
    
            hidden = transformer_model(hidden)[:, -1, :].unsqueeze(1).view(B, C, 1, H, W)
    
            pred = auto_encoder.decode(hidden)'''

        if recon_loss<.01:
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(test[0,0,0,:,:].detach().numpy())
            # plt.title("Image")
            # plt.subplot(122)
            # plt.imshow(recon[0,0,0,:,:].detach().numpy())
            # plt.title("Reconstruction")
            # plt.show()
            plt.figure()
            plt.subplot(121)
            plt.imshow(labels[0, 0, 0, :, :].detach().numpy())
            plt.title("label")
            plt.subplot(122)
            plt.imshow(pred[0, 0, 0, :, :].detach().numpy())
            plt.title("prediction")
            plt.show()





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








