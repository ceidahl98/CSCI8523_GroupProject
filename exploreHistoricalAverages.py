from netCDF4 import Dataset,MFDataset
from historical_averages import HistoricalAverages
import torchvision.transforms as transforms
import torch
import numpy as np

files = [r"data\1981_1985.nc", r"data\1986_1990.nc",
             r"data\1991_1995.nc", r"data\1996_2000.nc",
             r"data\2001_2005.nc", r"data\2006_2010.nc",
             r"data\2011_2015.nc", r"data\2016_2020.nc",
             r"data\2021_2024.nc"]
file = r"data\2011_2015.nc"
data = None
with open("historical_averages.npy","rb") as fp:
    data = np.load(fp)


# 640        256
dataset = Dataset(file).variables['sst']

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.38, 0.332)
    ])

HA = HistoricalAverages(data,t_window=1,lat_window=16,lon_window=16,transform=transform) 
lats = [416, 416, 416, 416]
lons = [ 896,  896,  896,  896]
times = [4,5,6,7]

averages = HA.get_historical_averages(times,lats,lons)

averages_temp = averages*0.332+0.38
print(averages.shape)
print(averages)
print(len(averages))

