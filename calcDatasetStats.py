from netCDF4 import Dataset, MFDataset
import numpy as np
from tqdm import tqdm


files = [
    r"D:\Datasets\data\1981_1985.nc", r"D:\Datasets\data\1986_1990.nc",
    r"D:\Datasets\data\1991_1995.nc", r"D:\Datasets\data\1996_2000.nc",
    r"D:\Datasets\data\2001-2005.nc", r"D:\Datasets\data\2006-2010.nc",
    r"D:\Datasets\data\2011-2015.nc", r"D:\Datasets\data\2016-2020.nc",
    r"D:\Datasets\data\2021-2024.nc"
]


dataset = MFDataset(files)
sst_data = dataset.variables['sst']


total_sum = 0
total_count = 0
total_sum_of_squares = 0


for t in tqdm(range(sst_data.shape[0]), desc="Processing time slices"):

    sst_slice = sst_data[t, :, :]
    valid_data = sst_slice[sst_slice > -100]  # Mask land/invalid values


    total_sum += valid_data.sum()
    total_count += valid_data.size
    total_sum_of_squares += (valid_data ** 2).sum()

# Final calculations
mean = total_sum / total_count
variance = (total_sum_of_squares / total_count) - (mean ** 2)
std = np.sqrt(variance)

print(f"Mean: {mean}, Std: {std}")
