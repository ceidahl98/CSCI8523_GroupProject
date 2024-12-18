import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def normalize_with_mask(tensor, mask):
    tensor_min = tensor[mask].min()
    tensor_max = tensor[mask].max()
    tensor[mask] = (tensor[mask] - tensor_min) / (tensor_max - tensor_min)
    tensor[~mask] = .5
    return tensor, tensor_min, tensor_max

def lat_to_index_northup(degree):
    if degree>-90 and degree<90:
        index = degree*4//1+720/2
    return int(index)

def lon_to_index_northup(degree):
    if degree>-180 and degree<180:
        if degree>=0:
            index = degree*4//1
        else:
            index = 1440+degree*4//1
    return int(index)

def lon_to_index_inverted(degree):
    #Same as regular
    if degree>-180 and degree<180:
        if degree>=0:
            index = degree*4//1
        else:
            index = 1440+degree*4//1
    return int(index)

def lat_to_index_inverted(degree):
    index = degree*4//1+720/2
    return int(index)

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

        averages,norm_min,norm_max = normalize_with_mask(averages,ocean_mask)


        if self.transform:
            print(averages.shape)
            print(self.transform(averages).shape)
            averages = self.transform(averages).permute(1,0,2)
        return averages, norm_min, norm_max
    
    def get_historical_average_temps(self,times,lats,lons):
        # Need to check what format the lats,lons,times are when getting them from SST dataset
        lat = lats[0]
        lon = lons[0]
        print(lat,lon)
        averages = self.dataset[times,lat:lat+self.lat_window,lon:lon+self.lon_window]     #Verify how this pulls from it when home, also may need to convert times to [0-365]

        ocean_mask = averages > -100
        # labels_mask = averages > -100
        '''image[image < -100] = np.mean(image[ocean_mask])

        label[label < -100] = np.mean(label[labels_mask])'''

        averages[~ocean_mask] = np.nan
        return averages
    
    def plot_globe_righted(self,day=0,boxes=None,colors=None):
        
        
        averages = self.dataset[0,::-1,:] #Flip so graph is north facing up
        averages[averages<-100] = np.nan
        
        x = 720
        y=1440
        plt.figure()
        plt.imshow(averages)

        if boxes is not None:
            if colors is None:
                colors=['r']*len(boxes)
            ax = plt.gca()
            for box,color in zip(boxes,colors):
                x1, y1 = box
                ax.add_patch(Rectangle((lon_to_index_northup(y1),x-lat_to_index_northup(x1)),self.lat_window,self.lon_window,edgecolor=color,fill=False))
        plt.show()

    def plot_globe(self,day=0,boxes=None,colors=None):
        averages = self.dataset[0,:,:] #Show how dataset is stored
        averages[averages<-100] = np.nan
        
        x = 720
        y=1440
        plt.figure()
        plt.imshow(averages)

        if boxes is not None:
            if colors is None:
                colors=['r']*len(boxes)
            ax = plt.gca()
            for box,color in zip(boxes,colors):
                x1, y1 = box
                ax.add_patch(Rectangle((lon_to_index_northup(y1),lat_to_index_northup(x1)-48),self.lat_window,self.lon_window,edgecolor=color,fill=False))
        plt.show()

if __name__=="__main__":
    degree = 28
    print(lat_to_index_northup(degree))