'''
Load and evaluate the model
'''
import config
import utils
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt

def normalize_images(*images):
    """Normalize image data to [0, 1] range."""
    normalized_images = []
    for image in images:    
        if image.min() < 0:  # Assuming images are in [-1, 1]
            normalized_images.append((image + 1) / 2)  # Normalize to [0, 1]
        elif image.max() > 1:  # Assuming images are in [0, 255]
            normalized_images.append(image / 255.0)  # Normalize to [0, 1]
        else:
            normalized_images.append(image) # Assuming images are already in [0, 1]
    return normalized_images

def eval_dataset(generator, generator_optimizer, discriminator,
                 discriminator_optimizer, dataset, savepath):
    # Load the model
    utils.load_checkpoint(generator, generator_optimizer,
                          config.CHECKPOINT_GEN, config.LEARNING_RATE)
    utils.load_checkpoint(discriminator, discriminator_optimizer,
                          config.CHECKPOINT_DISC, config.LEARNING_RATE)

    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Generate and save the images
    figure, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, (x, y) in enumerate(loader):
        if i >= 3:  # Only process the first 3 images
            break
        
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        with torch.no_grad():
            y_fake = generator(x)
            
            y_fake, x, y = normalize_images(y_fake, x, y)
            
            
            # First row: Input images
            axes[0, i].imshow(x[0].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title("Input" if i == 0 else "")
            axes[0, i].axis("off")
            
            # Second row: Generated images
            axes[1, i].imshow(y_fake[0].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title("Generated" if i == 0 else "")
            axes[1, i].axis("off")
            
            # Third row: Target images
            axes[2, i].imshow(y[0].permute(1, 2, 0).cpu().numpy())
            axes[2, i].set_title("Target" if i == 0 else "")
            axes[2, i].axis("off")

    plt.savefig(savepath)
    plt.close(figure)  # Close the figure to free memory



if __name__ == "__main__":
    from discriminator import Discriminator
    from generator import Generator
    from torch import optim, nn
    from dataset import Dataset

    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()  # Binary Cross Entropy
    L1_LOSS = nn.L1Loss()  # L1 loss

    dataset = Dataset(
        image_globbing_pattern=r"D:\CZI_scope\code\data\videos\training_data\stitched\*.jpg",
        target_globbing_pattern=r"D:\CZI_scope\code\data\videos\training_data\stitched\*.jpg",
        make_even=False, make_square=False, match_shape=False,
        target_input_combined=True, axis="x",
        transform=(None, config.transform_only_input, config.
                   transform_only_target))
    # split the dataset into train and validation sets
    train_len = int(len(dataset)*0.8)  # Use 80% of the dataset for training
    val_len = len(dataset) - train_len  # Use the remaining 20% for validation
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_len, val_len])
    
    eval_dataset(gen, opt_gen, disc, opt_disc, val_set, r"D:\CZI_scope\code\ml_models\pix2pix\eval")
