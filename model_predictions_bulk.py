import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from netCDF4 import Dataset,MFDataset
import matplotlib.pyplot as plt
import pandas as pd
from EncoderDecoder import autoEncoder
from transformer import GPT, GPTConfig
import csv
from scipy.interpolate import griddata

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def interpolate_map(data, mask, region_size, interpolation_factor=2):
    x, y = np.meshgrid(np.arange(region_size), np.arange(region_size))
    valid_points = mask > 0  # Use only valid (non-land) points
    points = np.column_stack((x[valid_points], y[valid_points]))
    values = data[valid_points]

    # Create a finer grid
    new_size = region_size * interpolation_factor
    finer_x, finer_y = np.meshgrid(
        np.linspace(0, region_size - 1, new_size),
        np.linspace(0, region_size - 1, new_size)
    )

    # Interpolate the data
    interpolated_data = griddata(points, values, (finer_x, finer_y), method='linear')
    return interpolated_data
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
        normalized_tensor[~mask] = -.1
        return normalized_tensor

    def unnormalize_with_mask(self, tensor, mask, min_val=-1.8, max_val=38.58):
        """Unnormalize tensor back to the original scale, applying mask."""
        unnormalized_tensor = tensor.copy()
        unnormalized_tensor[mask] = unnormalized_tensor[mask] * (max_val - min_val) + min_val
        unnormalized_tensor[~mask] = np.nan  # Set masked values to NaN
        return unnormalized_tensor

    def convert_coordinates(self, lat, lon):

        return lat, lon

    def get_item(self, t, lat, lon):
        #lat, lon = self.convert_coordinates(lat, lon)
        image = self.dataset[t:t + self.t_window, lat:lat + self.lat_window, lon:lon + self.lon_window]
        ocean_mask = image > -100
        image = self.normalize_with_mask(image, ocean_mask)
        image = self.transform(image).permute(1, 0, 2).unsqueeze(0)
        lats = torch.tensor(lat, device=self.device).view(1,)
        lons = torch.tensor(lon, device=self.device).view(1,)
        return image.to(self.device), lats, lons

    def get_label(self, t, lat, lon):
        #lat, lon = self.convert_coordinates(lat, lon)
        image = self.dataset[t, lat:lat + self.lat_window, lon:lon + self.lon_window]

        ocean_mask = image > -100
        image = self.normalize_with_mask(image, ocean_mask)
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def predict(self, image, lats, lons, horizon):
        """Generate predictions for the given horizon."""
        #print("lats: ",lats," lons: ", lons)
        image = image.unsqueeze(0)  # Add batch dimension

        z = self.auto_encoder.encode(image)
        B, C, D, H, W = z.shape
        z = z.permute(0, 2, 3, 4, 1).flatten(start_dim=2)

        for _ in range(horizon):

            pred = self.transformer(z, lats, lons)[:, -1, :]  # Get the last token prediction

            z = torch.cat([z, pred.unsqueeze(1)], dim=1)[:, -4:, :]  # Keep the last 4 tokens

        pred = self.auto_encoder.decode(z[:, -1, :].view(B, C, 1, H, W)).squeeze(0)
        return pred

    def pixelwise_mse_loss(self, pred, label, mask):
        """Compute pixel-wise MSE error map, masked for ocean values only."""
        squared_error = (pred - label) ** 2


        masked_error = squared_error * mask  # Apply the ocean mask

        return masked_error

    def get_patch(self, t, lat, lon):
        # lat, lon = self.convert_coordinates(lat, lon)
        # lat = max(0, min(lat, self.dataset.shape[1] - self.lat_window))
        # lon = max(0, min(lon, self.dataset.shape[2] - self.lon_window))
        image = self.dataset[t:t + self.t_window, lat:lat + self.lat_window, lon:lon + self.lon_window]
        ocean_mask = image > -100
        image = self.normalize_with_mask(image, ocean_mask)
        return self.transform(image).permute(1, 0, 2).unsqueeze(0), torch.tensor(ocean_mask, dtype=torch.float32)


    def calculate_rmse(self, coords_dict, horizons=[1,2,5,10]):
        """Calculate RMSE for regions with variable sizes and output pixel-wise error maps."""
        results = []
        region_sizes = {
            "Gulf of Mexico": 64,
            "Bohai Sea": 64,
            "Shallow Atlantic": 32,
            "Deep Atlantic": 32,
            "Shallow Pacific": 32,
            "Deep Pacific": 32
        }
        historicalAverages = np.load('./historical_averages.npy')
        print(historicalAverages.shape,"HA")

        for horizon in horizons:
            print(f"Processing horizon: {horizon}")
            horizon_results = []

            for region, coords_list in coords_dict.items():
                print(f"Processing region: {region}")
                region_size = region_sizes[region]  # Get region size dynamically
                error_map = np.zeros((region_size, region_size))
                per_error_map = np.zeros((region_size, region_size))
                avg_error_map= np.zeros((region_size, region_size))
                patch_avg_map = np.zeros((region_size, region_size))
                prediction_map = np.zeros((region_size, region_size))
                ground_truth_map = np.zeros((region_size, region_size))
                ocean_mask_map = np.zeros((region_size, region_size))
                patch_size = 16  # Each patch size
                patches_per_dim = region_size // patch_size

                start_lat, start_lon = coords_list[0]  # Starting latitude and longitude
                current_ha = historicalAverages
                print(current_ha.shape,"CURRENT")
                for pi in range(patches_per_dim):
                    for pj in range(patches_per_dim):
                        lat = int(start_lat - (patches_per_dim - 1 - pi) * patch_size)
                        lon = int(start_lon + (pj * patch_size))
                        patch_errors = np.zeros((patch_size, patch_size))
                        avg_patch_errors = np.zeros((patch_size, patch_size))
                        per_patch_errors = np.zeros((patch_size, patch_size))
                        patch_predictions = np.zeros((patch_size, patch_size))
                        patch_mask_sum = np.zeros((patch_size, patch_size))
                        patch_ground_truth = np.zeros((patch_size, patch_size))
                        patch_avg = np.zeros((patch_size, patch_size))
                        combined_mask = np.zeros((patch_size, patch_size))

                        for t in range(self.dataset.shape[0] - horizon):
                            patch, mask = self.get_patch(t, lat, lon)
                            patch = patch.to(self.device)
                            mask = mask.to(self.device)
                            label, _ = self.get_patch(t + horizon, lat, lon)
                            label = label[:,-1,:,:].to(self.device)
                            persistence, per_mask = self.get_patch(t,lat,lon)
                            persistence = persistence[:,-1,:,:]
                            per_mask = per_mask[-1,:,:]
                            ha = current_ha[t,lat:lat+self.lat_window,lon:lon+self.lon_window]

                            ha_mask = ha != np.nan
                            ha = torch.tensor(self.normalize_with_mask(ha,ha_mask),device=device)
                            pred = self.predict(patch, torch.tensor(lat).view(1,).to(self.device), torch.tensor(lon).view(1,).to(self.device), horizon)

                            mask = mask[-1,:,:]
                            mse = self.pixelwise_mse_loss(pred, label, mask).squeeze().cpu().numpy()
                            per_mse = self.pixelwise_mse_loss(torch.tensor(persistence,device=device), label, mask).squeeze().cpu().numpy()
                            avg_mse = self.pixelwise_mse_loss(torch.tensor(ha,device=device),label,mask).squeeze().cpu().numpy()

                            patch_errors += mse
                            avg_patch_errors+=avg_mse
                            per_patch_errors+=per_mse
                            patch_predictions += pred.squeeze().cpu().numpy().T * mask.squeeze().cpu().numpy()

                            patch_mask_sum += mask.squeeze().cpu().numpy()
                            patch_ground_truth += label.squeeze().cpu().numpy().T * mask.squeeze().cpu().numpy()
                            combined_mask += mask.squeeze().cpu().numpy()
                            patch_avg += ha.squeeze().cpu().numpy() *mask.squeeze().cpu().numpy()
                        # Avoid division by zero
                        patch_errors = np.divide(patch_errors, patch_mask_sum, where=patch_mask_sum != 0)
                        patch_errors = np.sqrt(patch_errors)
                        avg_patch_errors = np.divide(avg_patch_errors, patch_mask_sum, where=patch_mask_sum != 0)
                        avg_patch_errors = np.sqrt(avg_patch_errors)
                        per_patch_errors = np.divide(per_patch_errors, patch_mask_sum, where=patch_mask_sum != 0)
                        per_patch_errors = np.sqrt(per_patch_errors)
                        patch_predictions = np.divide(patch_predictions, patch_mask_sum, where=patch_mask_sum != 0)
                        patch_ground_truth= np.divide(patch_ground_truth, patch_mask_sum, where=patch_mask_sum != 0)
                        patch_avg = np.divide(patch_avg, patch_mask_sum, where=patch_mask_sum != 0)
                        error_map[pi * patch_size:(pi + 1) * patch_size, pj * patch_size:(pj + 1) * patch_size] = patch_errors
                        avg_error_map[pi * patch_size:(pi + 1) * patch_size,pj * patch_size:(pj + 1) * patch_size] = avg_patch_errors
                        per_error_map[pi * patch_size:(pi + 1) * patch_size,pj * patch_size:(pj + 1) * patch_size] = per_patch_errors
                        prediction_map[pi * patch_size:(pi + 1) * patch_size,pj * patch_size:(pj + 1) * patch_size] = patch_predictions
                        ground_truth_map[pi * patch_size:(pi + 1) * patch_size,pj * patch_size:(pj + 1) * patch_size] = patch_ground_truth
                        patch_avg_map[pi * patch_size:(pi + 1) * patch_size,pj * patch_size:(pj + 1) * patch_size] = patch_avg
                        ocean_mask_map[pi * patch_size:(pi + 1) * patch_size, pj * patch_size:(pj + 1) * patch_size] = combined_mask

                    #prediction_map = self.unnormalize_with_mask(prediction_map, ocean_mask_map > 0)
                    #ground_truth_map = self.unnormalize_with_mask(ground_truth_map, ocean_mask_map > 0)

                    #ocean_mask_map[ocean_mask_map <= 0] = np.nan
                    #error_map[ocean_mask_map <= 0] = np.nan  # Set land values to NaN for clear visualization
                    prediction_map[ocean_mask_map <= 0] = np.nan
                    ground_truth_map[ocean_mask_map <= 0] = np.nan
                    avg_error_map[ocean_mask_map <= 0] = np.nan
                    patch_avg_map[ocean_mask_map <= 0] = np.nan
                    per_error_map[ocean_mask_map <= 0] = np.nan
                    rmse_map = np.sqrt((prediction_map - ground_truth_map) ** 2)
                    rmse_map[ocean_mask_map <= 0] = np.nan  # Apply the mask to RMSE map
                    # Plot RMSE error map for the region
                    # Plot RMSE error map for the region
                    plt.figure(figsize=(8, 6))
                    plt.imshow(rmse_map, origin='lower', cmap='coolwarm')
                    plt.colorbar(label='RMSE')
                    plt.title(f"{region_size}x{region_size} Prediction RMSE Error Map for {region} (Horizon {horizon})")
                    plt.savefig(f"Pred_rmse_error_map_{region.replace(' ', '_')}_horizon_{horizon}.png")
                    plt.close()

                    plt.figure(figsize=(8, 6))
                    plt.imshow(avg_error_map, origin='lower', cmap='coolwarm')
                    plt.colorbar(label='RMSE')
                    plt.title(f"{region_size}x{region_size} Historical Average RMSE Error Map for {region} (Horizon {horizon})")
                    plt.savefig(f"Historical_Average_rmse_error_map_{region.replace(' ', '_')}_horizon_{horizon}.png")
                    plt.close()

                    plt.figure(figsize=(8, 6))
                    plt.imshow(per_error_map, origin='lower', cmap='coolwarm')
                    plt.colorbar(label='RMSE')
                    plt.title(
                        f"{region_size}x{region_size} Persistance RMSE Error Map for {region} (Horizon {horizon})")
                    plt.savefig(f"Persistance_rmse_error_map_{region.replace(' ', '_')}_horizon_{horizon}.png")
                    plt.close()

                    plt.figure(figsize=(8, 6))
                    plt.imshow(patch_avg_map, origin='lower', cmap='coolwarm')
                    plt.colorbar(label='Historical Average')
                    plt.title(f"{region_size}x{region_size} Historical Average Values (Normalized) for {region} (Horizon {horizon})")
                    plt.savefig(f"Historical_Average_map_{region.replace(' ', '_')}_horizon_{horizon}.png")
                    plt.close()

                    # Plot Prediction map for the region
                    plt.figure(figsize=(8, 6))
                    plt.imshow(prediction_map, origin='lower', cmap='coolwarm')
                    plt.colorbar(label='Predicted Values (Normalized)')
                    plt.title(f"{region_size}x{region_size} Prediction Map for {region} (Horizon {horizon})")
                    plt.savefig(f"prediction_map_{region.replace(' ', '_')}_horizon_{horizon}.png")
                    plt.close()

                    # Plot Ground Truth map for the region
                    plt.figure(figsize=(8, 6))
                    plt.imshow(ground_truth_map, origin='lower', cmap='coolwarm')
                    plt.colorbar(label='Ground Truth Values (Normalized)')
                    plt.title(f"{region_size}x{region_size} Ground Truth Map for {region} (Horizon {horizon})")
                    plt.savefig(f"ground_truth_map_{region.replace(' ', '_')}_horizon_{horizon}.png")
                    plt.close()

                    # Plot Combined Ocean Mask map for the region
                    plt.figure(figsize=(8, 6))
                    plt.imshow(ocean_mask_map, origin='lower', cmap='coolwarm')
                    plt.colorbar(label='Ocean Mask (1=Ocean, NaN=Land)')
                    plt.title(f"{region_size}x{region_size} Ocean Mask for {region} (Horizon {horizon})")
                    plt.savefig(f"ocean_mask_map_{region.replace(' ', '_')}_horizon_{horizon}.png")
                    plt.close()

                horizon_results.append((region, horizon, start_lat, start_lon, np.nanmean(rmse_map),np.nanmean(avg_error_map) , np.nanmean(per_error_map)))

            results.extend(horizon_results)
        # Write results to a CSV file
        with open('rmse_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Region", "Horizon", "Latitude", "Longitude", "Pred RMSE", "Historical Average RMSE", "Persistance RMSE"])
            writer.writerows(results)
        print("RMSE, prediction, ground truth, and ocean mask maps saved for all horizons.")



# Example usage
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.38, 0.332)
])

auto_encoder = autoEncoder(in_channels=1, embedding_dim=512).to(device)
transformer_model = GPT(GPTConfig, 2048).to(device)

state_dict = torch.load("models_scheduled_sampling/Epoch_10.pt", map_location=device)
auto_encoder.load_state_dict(state_dict['auto_encoder_state_dict'])
transformer_model.load_state_dict(state_dict['transformer_model_state_dict'])
for p in auto_encoder.parameters():
    p.requires_grad = False
for p in transformer_model.parameters():
    p.requires_grad = False
auto_encoder.eval()
transformer_model.eval()

files = [
            "data/1981_1985.nc", "data/1986_1990.nc", "data/1991_1995.nc",
            "data/1996_2000.nc", "data/2001-2005.nc", "data/2006-2010.nc",
            "data/2011-2015.nc", "data/2016-2020.nc", "data/2021-2024.nc"
        ]

dataset = MFDataset(files).variables['sst'][-366:,:,:]
predictions = SSTPredictor(auto_encoder, transformer_model, dataset, transform,device=device)

coords_dict = {
    "Gulf of Mexico": [(480, 1052),(480,1068), (480,1084),(480,1096),(464,1052),(464,1068),(464,1084),(464,1096),(448,1052),(448,1068),(448,1084),(448,1096),(432,1052),(432,1068),(432,1084),(432,1096)],
    "Bohai Sea": [(524, 468),(524,484),(524,500),(524,516),(508, 468),(508,484),(508,500),(508,516),(492, 468),(492,484),(492,500),(492,516),(476, 468),(476,484),(476,500),(476,516)],
    "Shallow Atlantic": [(440, 1332),(440,1348),(424,1332),(424,1348)],
    "Deep Atlantic": [(440, 1276),(440,1292),(424, 1276),(424,1292)],
    "Shallow Pacific": [(376, 640),(376,656),(360, 640),(360,656)],
    "Deep Pacific": [(376, 712),(376, 728), (360, 712),(360, 728)]
}

predictions.calculate_rmse(coords_dict)


