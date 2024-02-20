import config
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

# Correct denormalization with a mean and stdev
# Values for `mean` and `stdev` should be your original model's pre-processing values
def denormalize(img, mean, stdev):
    return img * stdev + mean

def save_examples(gen, val_loader, epoch, folder, mean, stdev):
    # Mean and Stdev should be announced where your normalization values are consistent
    loader_iter = iter(val_loader)
    fnames, x, y = next(loader_iter)
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = torch.clamp(y_fake, 0, 1)  # Ensuring the final value is in the range of [0,1]
        
        # The tensors: Convert to CPU and detach from graph
        y_fake = y_fake.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

        # iterating through batch to save every image in the list
        for idx in range(x.shape[0]):
            file_name = fnames[idx].split('/')[-1].split('.')[0] # Extracting the filename
            sample_fake = np.squeeze(y_fake[idx])  
            image_fake = denormalize(sample_fake, mean, stdev)
            sample_x = np.squeeze(x[idx])
            image_x = denormalize(sample_x, mean, stdev)
            sample_y = np.squeeze(y[idx])
            image_y = denormalize(sample_y, mean, stdev)
            
            # Plot the images with a color map and color bar
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(image_x, cmap='inferno')
            ax[0].set_title('Input')
            ax[0].axis('off')
            ax[1].imshow(image_y, cmap='inferno')
            ax[1].set_title('Target')
            ax[1].axis('off')
            ax[2].imshow(image_fake, cmap='inferno')
            ax[2].set_title('Generated')
            ax[2].axis('off')
            plt.savefig(path.join(folder, f"img_e{epoch}_b{idx}_{file_name}.jpg"))
            

        
    
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
        