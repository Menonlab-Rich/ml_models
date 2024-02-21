import config
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

# Correct denormalization with a mean and stdev
# Values for `mean` and `stdev` should be your original model's pre-processing values
def denormalize(img, mean=0, stdev=0):
    if mean == 0 or stdev == 0:
        return (img + 1)/2 # denormalize the range to [0, 1] because of tanh
    return (img * stdev) + mean # denormalize using the mean and stdev

def save_examples(gen, val_loader, epoch, folder, mean, stdev):
    loader_iter = iter(val_loader)
    fnames, x, y = next(loader_iter)
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        logging.error(y_fake.shape)
        logging.error(f'max: {y_fake.max()}, min: {y_fake.min()}')
        y_fake = y_fake.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        logging.error(f'after np, max: {y_fake.max()}, min: {y_fake.min()}')

        for idx in range(x.shape[0]):
            file_name = fnames[idx].split('/')[-1].split('.')[0]
            sample_fake = np.squeeze(y_fake[idx], 0)
            sample_fake = np.transpose(sample_fake, (1, 2, 0))
            image_fake = denormalize(sample_fake, mean, stdev)
            
            sample_x = np.squeeze(x[idx], 0)
            sample_x = np.transpose(sample_x, (1, 2, 0))
            image_x = denormalize(sample_x, mean, stdev)
            
            sample_y = np.squeeze(y[idx], 0)
            sample_y = np.transpose(sample_y, (1, 2, 0))
            image_y = denormalize(sample_y, mean, stdev)
            
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(image_x, cmap='inferno', vmin=0, vmax=1)
            ax[0].set_title('Input')
            ax[0].axis('off')
            ax[1].imshow(image_y, cmap='inferno', vmin=0, vmax=1)
            ax[1].set_title('Target')
            ax[1].axis('off')
            cax = ax[2].imshow(image_fake, cmap='inferno', vmin=0, vmax=1)
            ax[2].set_title('Generated')
            ax[2].axis('off')
            fig.colorbar(cax)
            plt.savefig(os.path.join(folder, f"img_e{epoch}_b{idx}_{file_name}.jpg"))

    gen.train()
            

        
    
    gen.train()
    
def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    

def load_checkpoint(model, optimizer, filename, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # if we don't do this then lr might be different
    # because the optimizer might have a different state
    # than the one we saved.
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        