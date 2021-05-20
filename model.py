import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision
import tqdm



class Sampling(nn.Module):
    def __init__(self, means: torch.Tensor, logvars: torch.Tensor):
        super().__init__()
        self.mean = means
        self.logvar = logvars

    def forward(self):
        return torch.randn_like(self.logvar) * torch.exp(self.logvar/2) + self.mean



class Encoder(nn.Module):
    def __init__(self, codings_size : int = 20, inp_shape : tuple = (28,28))->None:
        super().__init__()
        self.codings_size = codings_size
        self.inp_shape = inp_shape
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(in_features=self.inp_shape[0]*self.inp_shape[1],out_features=150)
        self.linear_2 = nn.Linear(in_features=150,out_features=100)
        self.coding_mean = nn.Linear(in_features=100,out_features=self.codings_size)
        self.coding_logvar = nn.Linear(in_features=100,out_features=self.codings_size)
    def forward(self,x):
        x = self.flatten(x)
        x = F.selu(self.linear_1(x))
        x = F.selu(self.linear_2(x))
        coding_mean = self.coding_mean(x)
        coding_logvar = self.coding_logvar(x)
        codings = Sampling(means=coding_mean,logvars=coding_logvar)()
        return coding_mean,coding_logvar,codings


class Decoder(nn.Module):
    def __init__(self, codings_size : int = 20,  inp_shape : tuple = (28,28))->None:
        super().__init__()
        self.codings_size = codings_size
        self.inp_shape = inp_shape

        self.linear_decoder_1 = nn.Linear(in_features=self.codings_size,out_features=100)
        self.linear_decoder_2 = nn.Linear(in_features=100,out_features=150)
        self.linear_decoder_3 = nn.Linear(in_features=150,out_features=self.inp_shape[0]*self.inp_shape[1])

    def forward(self,x):
        x = F.selu(self.linear_decoder_1(x))
        x = F.selu(self.linear_decoder_2(x))
        x = torch.sigmoid(self.linear_decoder_3(x))
        return x


class VAE_Dense(nn.Module):
    def __init__(self, Encoder,Decoder)->None:
        super().__init__()
      
        self.Encoder = Encoder
        self.Decoder = Decoder


    def forward(self,x):
        mean,logvar,codings = self.Encoder(x)
        x = self.Decoder(codings)
        return x,mean,logvar

