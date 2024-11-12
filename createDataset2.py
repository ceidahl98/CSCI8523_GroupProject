import os
import torch
import numpy as np
from netCDF4 import MFDataset, Dataset
import time
import csv
from multiprocessing import Pool, cpu_count
start_time=time.time()
# Configuration Parameters
files = ["1986_2000/1981_1985.nc", "1986_2000/1986_1990.nc",
         "1986_2000/1991_1995.nc", "1986_2000/1996_2000.nc"]
batch_size = 16
lat_increment, lon_increment =20, 20
lat_window, lon_window = 100, 100
time_window = 4
chunk_size = 10000

dataset = MFDataset(files).variables["sst"]


lat_range = torch.arange(0, 720 - lat_window, lat_increment)
lon_range = torch.arange(0, 1440 - lon_window, lon_increment)
num_time = dataset.shape[0] - time_window
dataset = MFDataset(files).variables["sst"]
num_data_points = len(lat_range) * num_time * len(lon_range)
data_points = []
total=0
# Generate tuples for each data slice
for t in range(num_time):
    print(total/num_data_points*100, "Percent Complete")
    for lat in lat_range:
        for lon in lon_range:
            # Append tuple (time index, lat index, lon index, data array)
            data_points.append((t, lat.item(), lon.item()))
            total+=1


csv_file_path = "data_points.csv"
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write header (optional)
    writer.writerow(["Time_Index", "Lat_Index", "Lon_Index"])

    # Write each data point tuple
    writer.writerows(data_points)

print(time.time()-start_time,"Elapsed")