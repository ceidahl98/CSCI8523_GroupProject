import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import numpy as np
from netCDF4 import MFDataset,Dataset
import pandas as pd
from EncoderDecoder import autoEncoder
from transformer import GPT, GPTConfig
import csv
import argparse
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
import h5netcdf
global_dataset = None



def normalize_with_mask(tensor, mask):
    tensor_min = tensor[mask].min()
    tensor_max = tensor[mask].max()
    #print(tensor_max,"MAX")
    #print(tensor_min,"min")
    if tensor_max == tensor_min:
      tensor[mask] = 0
      tensor[~mask] = 0
    else:
      tensor[mask] = (tensor[mask] - tensor_min) / (tensor_max - tensor_min)
      tensor[~mask] = 0.5
    return tensor

class SSTDataset(torch.utils.data.Dataset):
    def __init__(self, data_point_csv, file_paths, lon_window, lat_window, t_window, transform=None, target_transform=None):
        self.data = data_point_csv
        self.file_paths = file_paths  # List of file paths
        self.t_window = t_window
        self.lon_window = lon_window
        self.lat_window = lat_window
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = None

    def __len__(self):
        return len(self.data)

    def _initialize_dataset(self):
        """
        Initialize MFDataset instance.
        """
        if self.dataset is None:
            self.dataset = MFDataset(self.file_paths)
            self.sst = self.dataset.variables['sst']

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        t = row['Time_Index']
        lat = row['Lat_Index']
        lon = row['Lon_Index']

        # Determine which file to open based on time index

            # Initialize the HDF5 dataset for the worker if not already opened



        image = self.sst[t:t + self.t_window, lat:lat + self.lat_window, lon:lon + self.lon_window]
        label = self.sst[t + self.t_window + 1, lat:lat + self.lat_window, lon:lon + self.lon_window]

        # Process masks and normalization
        ocean_mask = image > -100
        labels_mask = label > -100
        image = normalize_with_mask(image, ocean_mask)
        label = normalize_with_mask(label, labels_mask)

        # Apply transformations
        if self.transform:
            image = self.transform(image).permute(1, 0, 2).unsqueeze(0)
        if self.target_transform:
            label = self.target_transform(label).unsqueeze(0)

        return image, label, torch.tensor(lat), torch.tensor(lon), torch.tensor(t)

    def get_file_index(self, t):
        # Map time index to the appropriate file and adjust time index to local file
        cumulative_sizes = self.cumulative_sizes
        for i, cum_size in enumerate(cumulative_sizes):
            if t < cum_size:
                local_t = t
                if i > 0:
                    local_t -= cumulative_sizes[i-1]
                #print(t,"T",local_t,"LOCAL")
                return i, local_t
        raise IndexError("Time index out of range")

files = ["data/1981_1985.nc", "data/1986_1990.nc",
             "data/1991_1995.nc", "data/1996_2000.nc",
             "data/2001-2005.nc", "data/2006-2010.nc",
             "data/2011-2015.nc", "data/2016-2020.nc",
             "data/2021-2024.nc"]


def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._initialize_dataset()


# Function to save the model
def save_model(auto_encoder, transformer_model, optimizer, epoch, loss, path):

        torch.save({
            'epoch': epoch,
            'auto_encoder_state_dict': auto_encoder.state_dict(),
            'transformer_model_state_dict': transformer_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"Model saved to {path}")

# Define the main function for multiprocessing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size per GPU')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='Number of nodes for distributed training')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Launch the training process
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # Initialize the process group
    # dist.init_process_group(backend='nccl', init_method=args.dist_url,
    #                         world_size=args.world_size, rank=args.gpu)

    # Set the device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')

    # Define your normalize functions and dataset class (as before)
    # ...

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.38, 0.332)
    ])

    # Dataset and DataLoader


    t_window = 4
    lon_window = 16
    lat_window = 16
    batch_size = args.batch_size
    num_epochs = args.epochs




    data_points = './data_points.csv'
    csv_ = pd.read_csv(data_points)
    total_len = int(csv_.shape[0])
    train_len = int(total_len * 38 / 40)
    val_len = int(total_len * 1 / 40)
    test_len = total_len - train_len - val_len
    train_csv = csv_.iloc[0:train_len, :]
    val_csv = csv_.iloc[train_len:train_len+val_len, :]
    test_csv = csv_.iloc[:-test_len, :]

    train_dataset = SSTDataset(train_csv, files, lon_window, lat_window, t_window,
                               transform=transform, target_transform=transform)
    val_dataset = SSTDataset(val_csv, files, lon_window, lat_window, t_window,
                             transform=transform, target_transform=transform)
    test_dataset = SSTDataset(test_csv, files, lon_window, lat_window, t_window,
                              transform=transform, target_transform=transform)


    # Define your dataset and DataLoader as before, but without DistributedSampler
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,worker_init_fn=worker_init_fn)

    # Initialize models
    in_channels = 1
    embedding_dim = 512

    auto_encoder = autoEncoder(in_channels, embedding_dim).to(device)
    transformer_model = GPT(GPTConfig, 2048).to(device)
    torch.compile(auto_encoder)
    torch.compile(transformer_model)
    # Define optimizer and loss function
    optim = torch.optim.AdamW(list(auto_encoder.parameters()) + list(transformer_model.parameters()),
                              lr=0.0002, betas=(0.9, 0.95), eps=1e-8)
    loss_fn = nn.MSELoss()
    l_pred = 1
    l_recon = 1
    model_save_iter = 0

    checkpoint_path = './models/checkpoint.pt'
    epoch_save_path = './models/'
    csv_file_path = './models/loss_csv.csv'
    # Training loop as before, but without DDP synchronization
    total_batches=len(train_loader)
    for epoch in range(num_epochs):
        print("training")
        auto_encoder.train()
        transformer_model.train()
        progress_bar = tqdm(iter(train_loader), total=total_batches, desc=f"Epoch [{epoch + 1}/{num_epochs}]",
                            leave=False)
        for image, labels, lats, lons, times in progress_bar:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lats = lats.to(device, non_blocking=True)
            lons = lons.to(device, non_blocking=True)

            optim.zero_grad()

            # Forward pass
            hidden = auto_encoder.encode(image)
            recon = auto_encoder.decode(hidden)
            B, C, D, H, W = hidden.shape

            hidden = hidden.permute(0, 2, 3, 4, 1).flatten(start_dim=2)
            hidden = transformer_model(hidden, lats, lons)[:, -1, :].unsqueeze(1).view(B, C, 1, H, W)

            pred = auto_encoder.decode(hidden)
            pred_loss = loss_fn(pred, labels)
            recon_loss = loss_fn(recon, image)
            loss = pred_loss + recon_loss
            progress_bar.set_postfix(loss=loss.item())
            # Backward and optimize
            loss.backward()
            optim.step()

            # Validation and testing loops as before

            model_save_iter += 1
            if model_save_iter % 10 == 0:
                save_model(auto_encoder, transformer_model, optim, epoch, loss.item(), checkpoint_path)

        # Validation loop (only main process)

        auto_encoder.eval()
        transformer_model.eval()
        val_loss = 0.0
        test_loss = 0.0

        with torch.no_grad():
            for image, labels, lats, lons, times in val_loader:
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lats = lats.to(device, non_blocking=True)
                lons = lons.to(device, non_blocking=True)

                hidden = auto_encoder.module.encode(image)
                recon = auto_encoder.module.decode(hidden)
                B, C, D, H, W = hidden.shape

                hidden = hidden.permute(0, 2, 3, 4, 1).flatten(start_dim=2)
                hidden = transformer_model(hidden, lats, lons)[:, -1, :].unsqueeze(1).view(B, C, 1, H, W)

                pred = auto_encoder.module.decode(hidden)
                pred_loss = loss_fn(pred, labels)
                recon_loss = loss_fn(recon, image)
                val_loss += l_pred * pred_loss.item() + l_recon * recon_loss.item()

            for image, labels, lats, lons, times in test_loader:
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lats = lats.to(device, non_blocking=True)
                lons = lons.to(device, non_blocking=True)

                hidden = auto_encoder.module.encode(image)
                recon = auto_encoder.module.decode(hidden)
                B, C, D, H, W = hidden.shape

                hidden = hidden.permute(0, 2, 3, 4, 1).flatten(start_dim=2)
                hidden = transformer_model(hidden, lats, lons)[:, -1, :].unsqueeze(1).view(B, C, 1, H, W)

                pred = auto_encoder.module.decode(hidden)
                pred_loss = loss_fn(pred, labels)
                recon_loss = loss_fn(recon, image)
                test_loss += l_pred * pred_loss.item() + l_recon * recon_loss.item()

            # Average losses
            val_loss /= len(val_loader)
            test_loss /= len(test_loader)

            # Save model and log losses every 5 epochs
            if epoch % 5 == 0:
                save_model(auto_encoder, transformer_model, optim, epoch, loss.item(),
                           os.path.join(epoch_save_path, f'Epoch_{epoch}.pt'))
                with open(csv_file_path, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([epoch, val_loss, test_loss])

if __name__ == '__main__':
    main()
