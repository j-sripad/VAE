import torch
import torch.optim as optim
from torchvision import transforms
import torchvision
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
    parser.add_argument('--save_model', default="model/model.pth", type=int, help='Model save path')

    # args parse
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    input_shape = tuple(args.input_shape)
    latent_dim = args.latent_dimension
    PATH = args.save_model

     #defining transforms
    transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    #selecting GPU if available
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

   
   
    
    
    
    #initializing encoder and decoder

    print("init Encoder and Decoder")
    Encoder = model.Encoder(codings_size=latent_dim,inp_shape=input_shape).to(device)
    Decoder = model.Decoder(codings_size=latent_dim,inp_shape=input_shape).to(device)

    #initializing model
    net = model.VAE_Dense(Encoder,Decoder).to(device)
    optimizer = optim.Adam(net.parameters())


    print("started training..")
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
        
            inputs, labels = data
            inputs = inputs.cuda()
            #skipping offset batches
            if len(inputs) < batch_size:
                continue
            optimizer.zero_grad()
            outputs,mean,logvar = net(inputs)
            latent_loss = loss.latent_loss_bce(outputs,inputs.view(batch_size,784),mean,logvar)
            latent_loss.backward()

            optimizer.step()
            running_loss += latent_loss.item()
        print("loss after epochs =  "+str(epoch),running_loss/(i*batch_size))

   
        torch.save(net.state_dict(), PATH)
    print('Finished Training')
