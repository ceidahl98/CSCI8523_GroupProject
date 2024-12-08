import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt

# if torch.cuda.is_available():
#     DEVICE=torch.device('cuda:0')
# elif torch.mps.is_available():
#     DEVICE=torch.device("mps")
# else:
#     DEVICE=torch.device("cpu")

DATA_NA_VAL = -9.96921e36

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder,self).__init__(**kwargs)
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=2,padding=1,padding_mode='zeros'),
                nn.ReLU(),
                nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,padding_mode='zeros'),
                nn.ReLU(),
                # nn.BatchNorm2d(16),
                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1,padding_mode='zeros'),
                nn.ReLU(),
                # nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,padding_mode='zeros')
    )
        
    def forward(self, X, *args):
        return self.encoder(X)

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=8,out_channels=4,kernel_size=3,stride=2,padding=1,output_padding=1)
    )

    def init_state():
        raise NotImplementedError
        
    def forward(self, X):
        return self.decoder(X)

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))        
        x = self.fc3(x)
        
        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, m*n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x)) 
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 1, self.m, self.n)
        return x



class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x




class koopmanAE(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = encoderNet(m, n, b, ALPHA = alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(m, n, b, ALPHA = alpha)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back


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