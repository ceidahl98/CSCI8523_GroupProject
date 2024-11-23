import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import numpy as np
from netCDF4 import MFDataset
import pandas as pd
from EncoderDecoder import autoEncoder
from transformer import GPT, GPTConfig
import csv
import argparse
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm

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




# Function to save the model
def save_model(auto_encoder, transformer_model, optimizer, epoch, loss, path):
    if dist.get_rank() == 0:  # Only save on the main process
        torch.save({
            'epoch': epoch,
            'auto_encoder_state_dict': auto_encoder.module.state_dict(),
            'transformer_model_state_dict': transformer_model.module.state_dict(),
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
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.gpu)

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
    files = [r"D:\Datasets\data\1981_1985.nc", r"D:\Datasets\data\1986_1990.nc",
             r"D:\Datasets\data\1991_1995.nc", r"D:\Datasets\data\1996_2000.nc",
             r"D:\Datasets\data\2001-2005.nc", r"D:\Datasets\data\2006-2010.nc",
             r"D:\Datasets\data\2011-2015.nc", r"D:\Datasets\data\2016-2020.nc",
             r"D:\Datasets\data\2021-2024.nc"]

    t_window = 4
    lon_window = 16
    lat_window = 16
    batch_size = args.batch_size
    num_epochs = args.epochs

    dataset = MFDataset(files).variables['sst']

    data_points = './data_points.csv'
    csv_ = pd.read_csv(data_points)
    total_len = int(csv_.shape[0])
    train_len = int(total_len * 38 / 40)
    val_len = int(total_len * 1 / 40)
    test_len = total_len - train_len - val_len
    train_csv = csv_.iloc[0:train_len, :]
    val_csv = csv_.iloc[train_len+1:train_len+val_len, :]
    test_csv = csv_.iloc[:-test_len, :]

    train_dataset = SSTDataset(train_csv, dataset, lon_window, lat_window, t_window,
                               transform=transform, target_transform=transform)
    val_dataset = SSTDataset(val_csv, dataset, lon_window, lat_window, t_window,
                             transform=transform, target_transform=transform)
    test_dataset = SSTDataset(test_csv, dataset, lon_window, lat_window, t_window,
                              transform=transform, target_transform=transform)

    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    # Initialize models
    in_channels = 1
    embedding_dim = 512

    auto_encoder = autoEncoder(in_channels, embedding_dim).to(device)
    transformer_model = GPT(GPTConfig, 2048).to(device)

    # Wrap models with DistributedDataParallel
    auto_encoder = nn.parallel.DistributedDataParallel(auto_encoder, device_ids=[args.gpu])
    transformer_model = nn.parallel.DistributedDataParallel(transformer_model, device_ids=[args.gpu])

    # Define optimizer and loss function
    optim = torch.optim.AdamW(list(auto_encoder.parameters()) + list(transformer_model.parameters()),
                              lr=0.0002, betas=(0.9, 0.95), eps=1e-8)
    loss_fn = nn.MSELoss()

    # Variables for saving models and logging
    l_pred = 1
    l_recon = 1
    model_save_iter = 0
    count = 0
    checkpoint_path = './models/checkpoint.pt'
    epoch_save_path = './models/'
    csv_file_path = './models/loss_csv.csv'

    # Only the main process should write to the CSV file
    if args.gpu == 0 and not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch", "Val_Loss", "Test_Loss"])

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle data differently every epoch
        auto_encoder.train()
        transformer_model.train()

        for image, labels, lats, lons, times in tqdm(iter(train_loader)):
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lats = lats.to(device, non_blocking=True)
            lons = lons.to(device, non_blocking=True)

            optim.zero_grad()
            count += 1

            # Forward pass
            hidden = auto_encoder.module.encode(image)  # (B, channels, time, features)
            recon = auto_encoder.module.decode(hidden)
            B, C, D, H, W = hidden.shape

            hidden = hidden.permute(0, 2, 3, 4, 1).flatten(start_dim=2)
            hidden = transformer_model(hidden, lats, lons)[:, -1, :].unsqueeze(1).view(B, C, 1, H, W)

            pred = auto_encoder.module.decode(hidden)
            pred_loss = loss_fn(pred, labels)
            recon_loss = loss_fn(recon, image)
            loss = l_pred * pred_loss + l_recon * recon_loss

            # Backward and optimize
            loss.backward()
            optim.step()

            model_save_iter += 1
            if model_save_iter % 10 == 0:
                save_model(auto_encoder, transformer_model, optim, epoch, loss.item(), checkpoint_path)

        # Validation loop (only main process)
        if dist.get_rank() == 0:
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
