import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset,MFDataset
import matplotlib.pyplot as plt
import time


start = time.time()
files = ["1986_2000/1981_1985.nc","1986_2000/1986_1990.nc","1986_2000/1991_1995.nc","1986_2000/1996_2000.nc"]

dataset = MFDataset(files).variables["sst"]

batch_size=16
lat_increment = 20
lon_increment = 20
lat_window = 100
lon_window =100
time_window = 4
num_years = 24




lat = torch.arange(0,720-lat_window,lat_increment)
lon = torch.arange(0,1440-lon_window,lon_increment)
num_lat = len(lat)
num_lon = len(lon)

timesteps = torch.arange(0,dataset.shape[0])
num_time = len(timesteps)-time_window

num_data_points = num_lat*num_time*num_lon

num_datasets = int(num_data_points/1000)
data_index = 0


New_Dataset = Dataset("./Datasets/FullDataset.nc",mode='w', format='NETCDF4')


New_Dataset.createDimension('lon',lon_window)
New_Dataset.createDimension('lat',lat_window)
New_Dataset.createDimension('t',time_window)
New_Dataset.createDimension('idx',num_data_points)
chunk_size = 1000
data_var = New_Dataset.createVariable('temperature',np.float32,('idx','t','lat','lon'),chunksizes=(1000,time_window,lat_window,lon_window),zlib=True)

num_batches = int(num_data_points/chunk_size)




current_t = 0
current_lat = 0
current_lon = 0
print(np.floor(40000/67*31))
current_location=0
for batch in range(num_batches):
    data_batch = []
    print("Percent Complete:", current_location/num_data_points*100)
    print("Time Elapsed(min): ", (time.time()-start)/60)
    start_location=current_location
    while current_location < start_location+chunk_size:
        data = dataset[current_t:current_t+time_window,current_lat*lat_increment:current_lat*lat_increment+lat_window,current_lon*lon_increment:current_lon*lon_increment+lon_window]
   
        data_batch.append(data)

        if current_location%67==0 and current_location!=0:
            current_lat+=1
            current_lon=0

        if current_location%(31*67) ==0 and current_location!=0:
            current_t+=1
            current_lat=0
        current_lon+=1
        current_location+=1

    data_batch = np.stack(data_batch,axis=0)
    data_var[batch*batch_size:batch*batch_size+chunk_size,:,:,:] = data_batch





New_Dataset.close()




