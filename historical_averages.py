import torch

def normalize_with_mask(tensor, mask):
    tensor_min = tensor[mask].min()
    tensor_max = tensor[mask].max()
    tensor[mask] = (tensor[mask] - tensor_min) / (tensor_max - tensor_min)
    tensor[~mask] = .5
    return tensor

#For extracting test results/visuals, we can probably do it locally if needed
class HistoricalAverages(torch.utils.data.Dataset):
    def __init__(self,dataset,t_window=1,lat_window=16,lon_window=16, transform = None):
        self.dataset = dataset
        self.transform = transform
        self.t_window = t_window
        self.lat_window = lat_window
        self.lon_window = lon_window

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def get_historical_averages(self,times,lats,lons):
        # Need to check what format the lats,lons,times are when getting them from SST dataset
        lat = lats[0]
        lon = lons[0]
        averages = self.dataset[times,lat:lat+self.lat_window,lon:lon+self.lon_window]     #Verify how this pulls from it when home, also may need to convert times to [0-365]

        ocean_mask = averages > -100
        # labels_mask = averages > -100
        '''image[image < -100] = np.mean(image[ocean_mask])

        label[label < -100] = np.mean(label[labels_mask])'''

        averages = normalize_with_mask(averages,ocean_mask)

        if self.transform:
            print(averages.shape)
            print(self.transform(averages).shape)
            averages = self.transform(averages).permute(1,0,2)
        return averages
