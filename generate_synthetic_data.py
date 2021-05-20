import argparse
import model
import torch
from torchvision.utils import save_image





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Synthetic Images')
    parser.add_argument('--input_shape',  nargs='+', type=int)
    parser.add_argument('--latent_dimension', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in to be generated')
    parser.add_argument('--model_path', default="model/model.pth", type=str, help='Path to the trained Model')
    parser.add_argument('--save_path', default="generated_images/", type=str, help='Path to store generated images')


     # args parse
    args = parser.parse_args()
    batch_size = args.batch_size
    input_shape = tuple(args.input_shape)
    latent_dim = args.latent_dimension
    PATH = args.model_path
    image_path = args.save_path
    #initializing encoder and decoder
    device = torch.device("cuda" if True else "cpu")
    print("init Encoder and Decoder")
    Encoder = model.Encoder(codings_size=latent_dim,inp_shape=input_shape).to(device)
    Decoder = model.Decoder(codings_size=latent_dim,inp_shape=input_shape).to(device)
   
    #initializing model
    net = model.VAE_Dense(Encoder,Decoder).to(device)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    with torch.no_grad():
        for i in range(batch_size):
            noise = torch.randn(1, 20).cuda()
            generated_image = Decoder(noise)
            generated_image = generated_image.view(input_shape)
            save_image(generated_image,image_path+str(i)+".png")




    