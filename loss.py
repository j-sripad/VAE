
import torch



def latent_loss_bce(pred, groundtruth, mean, log_var):
    bceloss = torch.nn.functional.binary_cross_entropy(pred, groundtruth, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return bceloss + KLD



def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD