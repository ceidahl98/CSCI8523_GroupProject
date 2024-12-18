from netCDF4 import Dataset,MFDataset
from historical_averages import HistoricalAverages
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt


def lat_to_index_inverted(degree):
    #Same as regular
    index = degree*4//1+720/2
    return int(index)

def lon_to_index_inverted(degree):
    if degree>-180 and degree<180:
        if degree>=0:
            index = degree*4//1
        else:
            index = 1440+degree*4//1
    return int(index)


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

HA = HistoricalAverages(data,t_window=1,lat_window=32,lon_window=32,transform=transform) 
lats = [416, 416, 416, 416]
lons = [ 896,  896,  896,  896]
times = [4,5,6,7]

averages,_,_ = HA.get_historical_averages(times,lats,lons)

averages_temp = averages*0.332+0.38

targets = [[30,-97],
           [41,117],
           [20,-27],
           [20,-41],
           [4,160],
           [4,178]]
colors = ['r','b','g','g','black','black']

HA.plot_globe(4,boxes=targets,colors=colors)
HA.plot_globe_righted(4,boxes=targets,colors=colors)

HA2 = HistoricalAverages(data,t_window=1,lat_window=48,lon_window=48,transform=transform) 
averages = HA2.get_historical_average_temps(range(366),[250],[700])

averages = averages.flatten()
# averages = averages*(norm_max-norm_min)+norm_min
plt.figure()
plt.plot(averages)
plt.show()

for item in targets:
    lat = item[0]
    lon=item[1]
    print(lat,lon)
    print(f"Lat Long: {item} --> index: {lat_to_index_inverted(lat)},{lon_to_index_inverted(lon)}")
    averages = HA2.get_historical_average_temps(1,[lat_to_index_inverted(lat)-48],[lon_to_index_inverted(lon)])
    plt.figure()
    plt.imshow(averages)
    plt.show()
