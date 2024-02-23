'''
Load and evaluate the model
'''
import config
import utils
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torch
from matplotlib import pyplot as plt
import numpy as np

def tensor_to_image(img: torch.tensor) -> np.ndarray:
    # switch the channel dimension to the last dimension
    img = torch.squeeze(img)
    img = img.cpu().detach().numpy()
    if img.shape[0] ==3:
        img = img.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
    return img

def denormalize_images(normalized_img, mean, std, max_pixel_value=255):
    """Denormalize image data from [-1, 1] range back to [0, 255]."""
    # Reverse the normalization formula
    img = (normalized_img * std + mean) * max_pixel_value
    # Ensure the pixel values are within the [0, 255] range
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def evaluate_consistency(dataset, savepath):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    fig, _ = generate_plot(loader, n_images=3, cmap='inferno')
    fig.savefig(savepath)
    
def generate_plot(loader, generator=None, n_images=3, cmap='viridis'):
    # Generate and save the images
    fig, axes = plt.subplots(3 if generator is not None else 2, n_images, figsize=(20, 20))
    for i, (_, x, y) in enumerate(loader):
        if i >= n_images:  # Only process the first 3 images
            break
        
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        with torch.no_grad():
            x = torch.tensor(x)
            y = torch.tensor(y)
            if generator is not None:
                y_fake = tensor_to_image(generator(x))
            x = tensor_to_image(x)
            y = tensor_to_image(y)
            
            
            # First row: Input images
            axes[0, i].imshow(x, vmin=0, vmax=1, cmap=cmap)
            axes[0, i].set_title("Input" if i == 0 else "")
            axes[0, i].axis("off")
            if generator is not None:
                # Second row: Generated images
                axes[1, i].imshow(y_fake, vmin=0, vmax=1, cmap=cmap)
                axes[1, i].set_title("Generated" if i == 0 else "")
                axes[1, i].axis("off")
                
                # Third row: Target images
                im = axes[2, i].imshow(y, vmin=0, vmax=1, cmap=cmap)
                axes[2, i].set_title("Target" if i == 0 else "")
                axes[2, i].axis("off")
            else:
                # Second row: Target images
                im = axes[1, i].imshow(y, vmin=0, vmax=1, cmap=cmap)
                axes[1, i].set_title("Target" if i == 0 else "")
                axes[1, i].axis("off")
            
    # add axis for colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
            
    return fig, axes
    

def eval_dataset(generator, generator_optimizer, discriminator,
                 discriminator_optimizer, dataset, savepath):
    # Load the model
    utils.load_checkpoint(generator, generator_optimizer,
                          config.CHECKPOINT_GEN, config.LEARNING_RATE)
    utils.load_checkpoint(discriminator, discriminator_optimizer,
                          config.CHECKPOINT_DISC, config.LEARNING_RATE)

    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    fig, axes = generate_plot(loader, savepath, generator, n_images=3)
    
    # Save the plot
    fig.savefig(savepath)
    
    



if __name__ == "__main__":
    from discriminator import Discriminator
    from generator import Generator
    from torch import optim, nn
    from dataset import Dataset

    # disc = Discriminator().to(config.DEVICE)
    # gen = Generator().to(config.DEVICE)
    # opt_disc = optim.Adam(
    #     disc.parameters(),
    #     lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    # opt_gen = optim.Adam(
    #     gen.parameters(),
    #     lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    # BCE = nn.BCEWithLogitsLoss()  # Binary Cross Entropy
    # L1_LOSS = nn.L1Loss()  # L1 loss

    # dataset = torch.load('/uufs/chpc.utah.edu/common/home/u0977428/ml_models/pix2pix/val.pt')
    # # split the dataset into train and validation sets
    # train_len = int(len(dataset)*0.8)  # Use 80% of the dataset for training
    # val_len = len(dataset) - train_len  # Use the remaining 20% for validation
    dataset = Dataset(
        image_globbing_pattern=r"D:\CZI_scope\code\data\videos\training_data\stitched\*.jpg",
        target_globbing_pattern=r"D:\CZI_scope\code\data\videos\training_data\stitched\*.jpg",
        target_input_combined=True,
        n_channels=1
    )
    
    #eval_dataset(gen, opt_gen, disc, opt_disc, val_set, r"/uufs/chpc.utah.edu/common/home/u0977428/ml_models/pix2pix/eval/eval.jpg")
    evaluate_consistency(dataset, r"D:\CZI_scope\code\ml_models\pix2pix\eval\eval_no_gen.jpg")
