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
    fname, x, y = next(iter(val_loader))
    x, y = x.to('cuda'), y.to('cuda')  # Assuming you're using CUDA
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
            # Given: Batch, 1 Channel, Width, and Height
            # Depending on your model's use, you'll focus on a small subset
            sample_fake = np.squeeze(y_fake[idx])  
            image_fake = denormalize(sample_fake, mean, stdev)
            
            plt.imsave(os.path.join(folder, f'e{epoch}_predicted_{fname[idx]}.png'), image_fake, cmap='inferno')
            

        
    
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
        
def denormalize(img, stdev, mean):
    img = np.array(img).astype(np.float32)
    return img * stdev + mean