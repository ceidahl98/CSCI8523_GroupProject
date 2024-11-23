import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE=torch.device('cuda:0')
else:
    DEVICE=torch.device("cpu")

DATA_NA_VAL = -9.96921e36

def Activation():
    return nn.SiLU() #swish

def Norm(in_size):
    return nn.GroupNorm(num_groups=32, num_channels=in_size, eps=1e-6, affine=True)

class resNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.2):
        super().__init__()


        self.resBlock = nn.Sequential(
            Norm(in_channels),
            Activation(),
            nn.Conv3d(
                in_channels,
                out_channels,
                groups=in_channels,
                kernel_size=(1,3,3),
                stride=1,
                padding=(0,1,1),
                bias=False,
            ),
            Norm(out_channels),
            Activation(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels,groups=out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.resBlock(x)


def downsample(in_channels,out_channels):
    return nn.Conv3d(in_channels,out_channels,groups=in_channels,kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))

def upsample(in_channels,out_channels):

    return nn.ConvTranspose3d(in_channels,
                              out_channels,
                              groups=out_channels,
                              kernel_size=(1,4,4),
                              stride=(1,2,2),
                              padding=(0,1,1))

class autoEncoder(nn.Module):
    def __init__(self,in_channels,embedding_dim):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.channels = [32,128,256,embedding_dim]


        self.in_conv = nn.Conv3d(in_channels, self.channels[0], groups=in_channels,kernel_size=(1,3,3), stride=1, padding=(0,1,1))
        self.encoder_layers.append(self.in_conv)



        self.out_conv = nn.Conv3d(self.channels[0],in_channels,groups=in_channels,kernel_size=(1,3,3), stride=1, padding=(0,1,1),bias=False)

        num_layers = len(self.channels)-1
        for i in range(num_layers):
            self.encoder_layers.append(
                nn.Sequential(
                    resNetBlock(self.channels[i],self.channels[i]),
                    resNetBlock(self.channels[i], self.channels[i]),
                    downsample(self.channels[i],self.channels[i+1])
                )
            )

        for i in range(4):
            self.encoder_layers.append(
                nn.Sequential(
                    resNetBlock(embedding_dim,embedding_dim)
                )
            )
            self.decoder_layers.append(
                nn.Sequential(
                    resNetBlock(embedding_dim,embedding_dim)
                )
            )
        for i in range(num_layers):
            self.decoder_layers.append(
                nn.Sequential(
                    upsample(self.channels[-(i+1)],self.channels[-(i+2)]),
                    resNetBlock(self.channels[-(i+2)],self.channels[-(i+2)]),
                    resNetBlock(self.channels[-(i + 2)], self.channels[-(i + 2)])
                )
            )
        self.decoder_layers.append(
            nn.Sequential(
                self.out_conv,
                #Activation()
            )
        )
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self,x):
        return self.encoder(x)

    def decode(self,z):
        return self.decoder(z)

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dropout = .1
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






























'''


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
    dataset.close()'''