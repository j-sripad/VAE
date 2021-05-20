import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision
import tqdm
import argparse
import model
import loss





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train VAELinear')
    parser.add_argument('--input_shape',  nargs='+', type=int)
    parser.add_argument('--latent_dimension', default=20, type=int)
    parser.add_argument('--Dataset', default="MNIST", type=str, help='MNIST or FashionMNIST')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    input_shape = tuple(args.input_shape)
    latent_dim = args.latent_dimension
    PATH = 'model/model.pth'
    transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    device = torch.device("cuda" if True else "cpu")

    if args.Dataset == 'FashionMNIST':
        train_set = torchvision.datasets.FashionMNIST('data', train=True, download=True,transform=transform)
        test_set = torchvision.datasets.FashionMNIST('data', train=False,download=True,transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set)
    else:
        train_set = torchvision.datasets.MNIST('data', train=True, download=True,transform=transform)
        test_set = torchvision.datasets.MNIST('data', train=False,download=True,transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set)

    #defining transforms
   
    
    
    
    #initializing encoder and decoder


    Encoder = model.Encoder(codings_size=latent_dim,inp_shape=(28,28)).to(device)
    Decoder = model.Decoder(codings_size=latent_dim,inp_shape=(28,28)).to(device)

    #initializing model
    net = model.VAE_Dense(Encoder,Decoder).to(device)
    optimizer = optim.Adam(net.parameters())



    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
        
            inputs, labels = data
            inputs = inputs.cuda()
            if len(inputs) < batch_size:
                continue
            optimizer.zero_grad()
            outputs,mean,logvar = net(inputs)
            latent_loss = loss.latent_loss_bce(inputs.view(batch_size,784), outputs,mean,logvar)
            latent_loss.backward()

            optimizer.step()
            running_loss += latent_loss.item()
        print("loss after epochs =  "+str(epoch),running_loss/(i*batch_size))

   
        torch.save(net.state_dict(), PATH)
    print('Finished Training')
