import os
import torch
import numpy as np
from netCDF4 import Dataset
import time
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# Function to check data quality within a single file
def checkData(args):
    t, lat, lon, file, time_window, lat_window, lon_window, time_offset = args
    dataset = Dataset(file).variables["sst"]  # Open the file within each worker process
    data = dataset[t:t + time_window, lat:lat + lat_window, lon:lon + lon_window]
    check = np.where(data < -10000, 1, 0)
    total_data = data.size

    if np.sum(check) < total_data * 0.3:  # If land is more than 30%, skip
        return t + time_offset, lat, lon  # Adjusted time index
    else:
        return None


# Function to process each NetCDF file independently
def process_file(file, csv_file_path, lat_window, lon_window, time_window, time_offset):
    # Define the ranges based on latitude and longitude window increments
    lat_increment, lon_increment = lat_window, lon_window
    lat_range = torch.arange(0, 720 - lat_window, lat_increment)
    lon_range = torch.arange(0, 1440 - lon_window, lon_increment)

    # Open the dataset and determine time range
    dataset = Dataset(file)
    num_time = dataset.variables["sst"].shape[0] - time_window - 1
    dataset.close()

    # Prepare arguments for multiprocessing
    all_combinations = [(t, lat.item(), lon.item(), file, time_window, lat_window, lon_window, time_offset)
                        for t in range(num_time) for lat in lat_range for lon in lon_range]

    # Process data and write results to CSV in append mode
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        chunk_size = 5000  # Adjust chunk size based on memory and processing requirements
        chunk = []

        with Pool(cpu_count() - 1) as pool:  # Leave one core free for system tasks
            for result in tqdm(pool.imap(checkData, all_combinations, chunksize=100), total=len(all_combinations),
                               desc=f"Processing {file}"):
                if result is not None:
                    chunk.append(result)

                # Write to CSV periodically when chunk is full
                if len(chunk) >= chunk_size:
                    writer.writerows(chunk)
                    chunk = []  # Clear the chunk after writing

            # Write any remaining data in the final chunk
            if chunk:
                writer.writerows(chunk)

    return num_time  # Return the number of time steps processed in this file


# Main function to process all files
def main():
    start_time = time.time()

    # Parameters
    files = [r"D:\Datasets\data\1981_1985.nc", r"D:\Datasets\data\1986_1990.nc",
             r"D:\Datasets\data\1991_1995.nc", r"D:\Datasets\data\1996_2000.nc",
             r"D:\Datasets\data\2001-2005.nc", r"D:\Datasets\data\2006-2010.nc",
             r"D:\Datasets\data\2011-2015.nc", r"D:\Datasets\data\2016-2020.nc",
             r"D:\Datasets\data\2021-2024.nc"]



    csv_file_path = "data_points.csv"
    lat_window, lon_window = 16, 16
    time_window = 4
    time_offset = 7041  # Initialize time offset

    # Initialize CSV file with header
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Time_Index", "Lat_Index", "Lon_Index"])  # Write header

    # Process each file independently, updating the time_offset for each file
    for file in files:
        num_time = process_file(file, csv_file_path, lat_window, lon_window, time_window, time_offset)
        time_offset += num_time  # Update the time offset for the next file

    print(f"{time.time() - start_time} seconds elapsed")


# Protect the entry point
if __name__ == "__main__":
    main()
