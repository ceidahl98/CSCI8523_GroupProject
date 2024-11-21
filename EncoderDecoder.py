import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE=torch.device('cuda:0')
elif torch.mps.is_available():
    DEVICE=torch.device("mps")
else:
    DEVICE=torch.device("cpu")

DATA_NA_VAL = -9.96921e36

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder,self).__init__(**kwargs)
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=1,padding=1,padding_mode='zeros'),
                nn.ReLU(),
                nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,padding_mode='zeros'),
                nn.ReLU(),
                nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1,padding_mode='zeros'),
                nn.ReLU(),
                nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1,padding_mode='zeros')
    )
        
    def forward(self, X, *args):
        return self.encoder(X)

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=8,out_channels=4,kernel_size=3,stride=1,padding=1)
    )

    def init_state():
        raise NotImplementedError
        
    def forward(self, X):
        return self.decoder(X)

if __name__=="__main__":
    files = ['sst.day.mean.1982.nc']
    dataset = Utils.extract_data(files)
    print(torch.tensor(dataset.variables['sst'][:].shape))
    data = torch.tensor(dataset.variables['sst'][40,:,:])
    # encoder = nn.Sequential(
    #             nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1,padding_mode='zeros'),
    #             nn.ReLU(),
    #             nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,padding_mode='zeros'),
    #             nn.ReLU(),
    #             nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1,padding_mode='zeros')
    # )
    encoder = Encoder()
    decoder = Decoder()

    print(data[data!=DATA_NA_VAL].min())
    data[data==DATA_NA_VAL] = np.nan
    data = data.flip(0)
    print(data.shape)
    print(data[None].shape)
    enc = encoder(data[None])
    print(enc.shape)
    
    dec = decoder(enc)
    print(dec.shape)

    plt.figure()
    plt.imshow(data)
    plt.show()
    dataset.close()