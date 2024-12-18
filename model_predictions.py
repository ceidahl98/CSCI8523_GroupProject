import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from netCDF4 import Dataset,MFDataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,Normalize
import pandas as pd
from EncoderDecoder import autoEncoder
from transformer import GPT, GPTConfig
import csv
from historical_averages import HistoricalAverages, lat_to_index_inverted, lon_to_index_inverted

def RMSE(X,Y,axis=(1,2)):
    lat_window=X.shape[1]
    lon_window=X.shape[2]
    return np.sqrt(np.nansum((Y-X)**2,axis=axis)/(lat_window*lon_window))

class SSTPredictor:
    def __init__(self, auto_encoder, transformer, dataset, transform, lat_window=16, lon_window=16, t_window=4, device="cpu"):
        self.auto_encoder = auto_encoder.to(device)
        self.transformer = transformer.to(device)
        self.dataset = dataset
        self.transform = transform
        self.lat_window = lat_window
        self.lon_window = lon_window
        self.t_window = t_window
        self.device = device

    def normalize_with_mask(self, tensor, mask, min_val=-1.8, max_val=38.58):

        normalized_tensor = tensor.copy()
        normalized_tensor[mask] = (normalized_tensor[mask] - min_val) / (max_val - min_val)
        normalized_tensor[~mask] = -0.1
        return normalized_tensor

    def get_item(self, t, lat, lon):

        image = self.dataset[t:t + self.t_window, lat:lat + self.lat_window, lon:lon + self.lon_window]
        ocean_mask = image > -100
        image = self.normalize_with_mask(image, ocean_mask)
        image = self.transform(image).permute(1, 0, 2).unsqueeze(0)
        lats = torch.tensor(lat, device=self.device).view(1,)
        lons = torch.tensor(lon, device=self.device).view(1,)
        return image.to(self.device), lats, lons

    def get_label(self, t, lat, lon):

        image = self.dataset[t, lat:lat + self.lat_window, lon:lon + self.lon_window]
        ocean_mask = image > -100
        image = self.normalize_with_mask(image, ocean_mask)
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def normalize_recon(self,x):
        x = x


    def predict(self, image, lats, lons, horizon, image_save=False):
        """Generate predictions for the given horizon."""
        predictions = []

        image = image.unsqueeze(0)  # Add batch dimension
        z = self.auto_encoder.encode(image)
        B, C, D, H, W = z.shape
        z = z.permute(0, 2, 3, 4, 1).flatten(start_dim=2)

        for t in range(horizon):
            pred = self.transformer(z, lats, lons)[:, -1, :]  # Get the last token prediction
            z = torch.cat([z, pred.unsqueeze(1)], dim=1)[:, -4:, :]  # Keep the last 4 tokens

            if image_save:
                predicted_image = self.auto_encoder.decode(z.view(B, C, 1, H, W))
                predictions.append(predicted_image)

        pred = self.auto_encoder.decode(z[:, -1, :].view(B, C, 1, H, W)).squeeze(0)
        return pred, predictions

    def evaluate(self, coords_list, horizon=1, image_save=False):

        for coords in coords_list:
            lat, lon = coords
            for t in range(self.dataset.shape[0] - horizon):
                image, lats, lons = self.get_item(t, lat, lon)
                label = self.get_label(t + horizon, lat, lon)
                pred, _ = self.predict(image, lats, lons, horizon, image_save=image_save)
                loss = nn.functional.mse_loss(pred,label)

                # Visualize
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(label[0,0,:,:].detach().cpu().numpy())
                axes[0].set_title("Ground Truth")
                axes[1].imshow(pred.squeeze(0).permute(1,2,0).detach().cpu().numpy())
                axes[1].set_title("Prediction")
                plt.show()

    def visualize_reconstruction(self, t, lat, lon, min_val=-1.8, max_val=38.58):
        # Get the input image and ground truth label
        image, _, _ = self.get_item(t, lat, lon)
        print(image.mean())
        image = image.unsqueeze(0)
        label = self.get_label(t, lat, lon)
        print(label.mean())

        # Reconstruct the image using the autoencoder
        reconstructed = self.auto_encoder.decode(self.auto_encoder.encode(image))[:,:,-1,:,:]
        print(reconstructed.shape)
        # Denormalize for visualization
        ocean_mask = label > -0.1  # Mask for valid regions
        #label_denorm = self.denormalize_with_mask(label.clone(), ocean_mask, min_val, max_val)
        #reconstructed_denorm = self.denormalize_with_mask(reconstructed.clone(), ocean_mask, min_val, max_val)

        # Visualize the ground truth and reconstructed image
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(label.squeeze(0).permute(1,2,0).cpu().detach().numpy())
        axes[0].set_title("Ground Truth")
        axes[1].imshow(reconstructed.view(1,16,16).permute(1,2,0).cpu().detach().numpy())
        axes[1].set_title("Reconstructed Image")
        plt.show()


torch.manual_seed(42) #makes training reproducible
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.38, 0.332)
    ])

in_channels = 1
embedding_dim = 512

auto_encoder = autoEncoder(in_channels, embedding_dim).to(device)
transformer_model = GPT(GPTConfig, 2048).to(device)

state_dict = torch.load("model/Epoch_15.pt",map_location=device)
auto_encoder.load_state_dict(state_dict['auto_encoder_state_dict'])
transformer_model.load_state_dict((state_dict['transformer_model_state_dict']))
auto_encoder.eval()
transformer_model.eval()

coords_list = [(424,720)] #change this to read from excel file, should be a list of tuples


horizon = 5


files = [r".\data\1981_1985.nc", r".\data\1986_1990.nc",
             r".\data\1991_1995.nc", r".\data\1996_2000.nc",
             r".\data\2001_2005.nc", r".\data\2006_2010.nc",
             r".\data\2011_2015.nc", r".\data\2016_2020.nc",
             r".\data\2021_2024.nc"]

averages = np.load("historical_averages.npy")
averages[averages<-100] = np.nan

lon=524
lat=468
lat_window = 48
lon_window = 48
t_window =4
batch_size=16

dataset = MFDataset(files).variables['sst'][-366:,:,:] #366 to match historical averages

rmse = RMSE(averages,dataset,axis=0)
mean = np.nanmean(averages-dataset,axis=0)

cmap = mpl.colormaps.get_cmap('bwr')  # viridis is the default colormap for imshow
# cmaplist = [cmap(i) for i in range(cmap.N)]
cmap.set_bad(color='black')
print(np.nanmin(mean),np.nanmax(mean),np.nanmean(mean))
norm = Normalize(vmin=-5,vmax=5)
# bounds = np.arange(np.min(mean),np.max(mean),.5)
# idx = np.searchsorted(bounds,0)
# bounds = np.insert(bounds,idx,0)
# norm = BoundaryNorm(bounds,cmap.N)

plt.figure()
plt.imshow(mean,cmap=cmap,norm=norm)
plt.show()




predictions = SSTPredictor(auto_encoder,transformer_model,dataset,transform)
HA_baseline = HistoricalAverages(averages,lat_window=lat_window,lon_window=lon_window,transform=transform)

#Get nov 12 through end of year (50 days)
averages1 = HA_baseline.get_historical_average_temps(np.arange(366-50,366),[lat],[lon])
averages2 = HA_baseline.get_historical_average_temps(np.arange(366-50),[lat],[lon]) #Improve reading in coords_list for historical averages, add option to get as temperature?
averages = np.concatenate([averages1,averages2])

region_data = dataset[:,lat:lat+lat_window,lon:lon+lon_window]

print(region_data.shape)
print(region_data.min(),region_data.max())
print(averages.shape)
print(averages.min(),averages.max())
plt.figure()
plt.plot(np.nanmean(averages,axis=(1,2)),'g')
plt.plot(np.mean(region_data,axis=(1,2)),'b')
plt.show()

rmse = RMSE(region_data,averages,axis=0)
print(rmse.shape)

plt.figure()
plt.imshow(rmse,cmap='bwr')
plt.show()

# for i in range(averages.shape[0]):
#     plt.figure()
#     plt.imshow(region_data[i,:,:]-averages[i,:,:],cmap='bwr')
#     plt.show()
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(region_data[i,:,:])
    # plt.subplot(122)
    # plt.imshow(averages[i,:,:])
    # plt.show()
targets = [[30,-97],
           [41,117],
           [20,-27],
           [20,-41],
           [4,160],
           [4,178]]
indeces = [[lat_to_index_inverted(lat)-48,lon_to_index_inverted(lon)] for lat,lon in targets]
print(indeces)

horizon = 2
for lat, long in indeces:
    averages1 = HA_baseline.get_historical_average_temps(np.arange(366-50,366),[lat],[long])
    averages2 = HA_baseline.get_historical_average_temps(np.arange(366-50),[lat],[long]) #Improve reading in coords_list for historical averages, add option to get as temperature?
    averages = np.concatenate([averages1,averages2])
    data = np.zeros((366-2*horizon,3,48,48))
    for i in range(horizon,366-horizon):    
        pred = np.zeros((48,48))
        label = np.zeros((48,48))
        for x in range(3):
            for y in range(3):
                images, lats, lons = predictions.get_item(0,lat+16*x,long+16*y)
                temp, preds = predictions.predict(images,lats,lons,horizon)
                data[i,1,16*x:16*(x+1),16*y:16*(y+1)] = temp.squeeze().cpu().detach().numpy()
                temp = predictions.get_label(i+horizon,lat+16*x,long+16*y)
                print(label)
                data[i,0,16*x:16*(x+1),16*y:16*(y+1)] = temp.squeeze().cpu().detach().numpy()
        plt.figure()
        plt.subplot(131)
        plt.imshow(data[i,0,:,:])
        plt.subplot(132)
        plt.imshow(data[i,1,:,:])
        plt.subplot(133)
        plt.imshow(averages[i+horizon,:,:])
        plt.show()
        if i>10:
            break

#predictions.visualize_reconstruction(0,424,720)

# predictions.evaluate(coords_list,horizon) #will alter this to output what we want to show after discussing




