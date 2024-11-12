import os
import torch
import numpy as np
from netCDF4 import MFDataset, Dataset
import time
from multiprocessing import Pool, cpu_count

# Configuration Parameters
files = ["1986_2000/1981_1985.nc", "1986_2000/1986_1990.nc",
         "1986_2000/1991_1995.nc", "1986_2000/1996_2000.nc"]
batch_size = 16
lat_increment, lon_increment = 20, 20
lat_window, lon_window = 100, 100
time_window = 4
chunk_size = 10000


def process_batch(args):
    batch_start, files = args  # Unpack arguments

    # Each process loads the dataset independently
    dataset = MFDataset(files).variables["sst"]
    current_t, current_lat, current_lon = 0, 0, 0
    current_location = batch_start
    data_batch = []

    # Generate data for batch
    for i in range(chunk_size):
        data = dataset[
               current_t:current_t + time_window,
               current_lat * lat_increment:current_lat * lat_increment + lat_window,
               current_lon * lon_increment:current_lon * lon_increment + lon_window
               ]
        data_batch.append(data)

        # Update indices
        if current_location % 67 == 0 and current_location != 0:
            current_lat += 1
            current_lon = 0
        if current_location % (31 * 67) == 0 and current_location != 0:
            current_t += 1
            current_lat = 0
        current_lon += 1
        current_location += 1

    # Write batch data to temporary file

    temp_file_path = f"D:/Datasets{batch_start}.nc"
    with Dataset(temp_file_path, mode='w', format='NETCDF4') as temp_ds:
        temp_ds.createDimension('lon', lon_window)
        temp_ds.createDimension('lat', lat_window)
        temp_ds.createDimension('t', time_window)
        temp_ds.createDimension('idx', chunk_size)
        temp_var = temp_ds.createVariable(
            'temperature', np.float32, ('idx', 't', 'lat', 'lon')
        )
        temp_var[:, :, :, :] = np.stack(data_batch, axis=0)
    return temp_file_path


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("D:/Datasets", exist_ok=True)

    # Load initial dataset for dimension calculation
    initial_dataset = MFDataset(files).variables["sst"]
    lat_range = torch.arange(0, 720 - lat_window, lat_increment)
    lon_range = torch.arange(0, 1440 - lon_window, lon_increment)
    num_lat, num_lon = len(lat_range), len(lon_range)
    num_time = initial_dataset.shape[0] - time_window
    num_data_points = num_lat * num_time * num_lon
    num_batches = num_data_points // chunk_size

    # Parallel processing
    start_time = time.time()
    with Pool(processes=4) as pool:
        batch_starts = [i * chunk_size for i in range(num_batches)]
        # Pass (batch_start, files) tuples to each worker
        temp_files = pool.map(process_batch, [(batch_start, files) for batch_start in batch_starts])

    # Consolidate all temporary files into the final dataset
    final_file = r"D:/Datasets\FullDataset.nc"
    with Dataset(final_file, mode='w', format='NETCDF4') as New_Dataset:
        New_Dataset.createDimension('lon', lon_window)
        New_Dataset.createDimension('lat', lat_window)
        New_Dataset.createDimension('t', time_window)
        New_Dataset.createDimension('idx', num_data_points)
        data_var = New_Dataset.createVariable(
            'temperature', np.float32, ('idx', 't', 'lat', 'lon')
        )

        # Append data from each temp file
        current_index = 0
        for temp_file in temp_files:
            with Dataset(temp_file, mode='r') as temp_ds:
                temp_data = temp_ds.variables['temperature'][:]
                data_var[current_index:current_index + chunk_size, :, :, :] = temp_data
                current_index += chunk_size

    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

    print(f"Total time taken: {(time.time() - start_time) / 60:.2f} minutes")