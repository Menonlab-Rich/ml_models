import config
import torch
import logging
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def generate_model_prediction(model, checkpoint_path, image_path, output_path, resize=1.):
    '''
    Save the predictions of a model on an image to a file
    
    Parameters:
    ----------
    model: torch.nn.Module
        Model to use for prediction
    checkpoint_path: str
        Path to the model checkpoint file
    image_path: str
        Path to the image file
    output_path: str
        Path to save the predictions to
    resize: float | tuple, Default: 1 (no resize)
        The size to resize the image to before predicting. 
        This should match the size the model was trained on
            If a float, the image is resized by the factor
            If a tuple, the image is resized to the specified size
    '''

    # Load the model from the specified path
    load_checkpoint(model, None, checkpoint_path) # optimizer is not needed for prediction
    # Load and preprocess the image
    image = Image.open(image_path)
    # check if resize is a float
    if isinstance(resize, float):
        image = image.resize((int(image.width * resize), int(image.height * resize)))
    elif isinstance(resize, tuple):
        image = image.resize(resize)
    else:
        try:
            resize = float(resize)
            image = image.resize((int(image.width * resize), int(image.height * resize)))
        except:
            raise ValueError("resize must be a float or a tuple of two integers")
    
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
    # Predict
    model.eval()
    with torch.no_grad():
        y_fake = model(torch.tensor(image_array, dtype=torch.float32))
        # Normalize y_fake from [0, 255] to [0, 1] for matplotlib
        y_fake = y_fake / 255.0
        suffix = output_path.split('.')[-1]
        # Save the predictions to a file
        if suffix == 'npy':
            np.save(output_path, y_fake.numpy())
        elif suffix in ['.jpg', '.jpeg', '.png']:
            # Convert to uint8 and save as an image
            Image.fromarray(y_fake.numpy()).save(output_path)



def save_examples(model, val_loader, epoch, folder, device):
    if not hasattr(save_examples, "fixed_samples"):
        accumulated_x, accumulated_y = [], []
        for batch in val_loader:
            batch_x, batch_y = batch[0].to(device), batch[1].to(device)
            accumulated_x.append(batch_x)
            accumulated_y.append(batch_y)
            if sum([x.shape[0] for x in accumulated_x]) >= 6:
                break
        x = torch.cat(accumulated_x, dim=0)[:6]
        y = torch.cat(accumulated_y, dim=0)[:6]
        save_examples.fixed_samples = (x, y)
    else:
        x, y = save_examples.fixed_samples

    model.eval()
    with torch.no_grad():
        y_fake = model(x)
        # Normalize y_fake from [0, 255] to [0, 1] for matplotlib
        y_fake = y_fake / 255.0
    
    # Assuming x is already normalized to [0, 1]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(6):
        row = i // 3
        col = i % 3
        # Display input (grayscale) image
        axs[row, col].imshow(x[i].cpu().squeeze().numpy(), cmap='gray', interpolation='nearest')
        axs[row, col].set_title(f"Input {i+1}")
        axs[row, col].axis('off')
        
        # Display predicted (RGB) image
        # Ensure y_fake is permuted from [C, H, W] to [H, W, C] for correct display
        axs[(row+1)%2, col].imshow(y_fake[i].cpu().detach().numpy().transpose(1, 2, 0))
        axs[(row+1)%2, col].set_title(f"Prediction {i+1}")
        axs[(row+1)%2, col].axis('off')

    plt.tight_layout()
    plt.savefig(f"{folder}/comparison_epoch_{epoch}.png")
    plt.close('all')
    
    model.train()


def save_checkpoint(model, optimizer, filename):
    '''
    Save the model and optimizer state to a file
    
    Parameters:
    ----------
    model: torch.nn.Module
        Model to save
    optimizer: torch.optim.Optimizer
        Optimizer to save
    filename: str
        Filename to save the model and optimizer state to
    '''
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def check_accuracy(loader, model, device="cuda", return_outputs=0):
    '''
    Compare the accuracy of the model using
    L2 loss and Correlation coefficient

    Parameters:
    ----------
    loader: torch.utils.data.DataLoader
        DataLoader containing the data to check accuracy on
    model: torch.nn.Module
        Model to check accuracy of
    device: str, Default: "cuda"
        Device to use for the model
    return_outputs: int, Default: 0
        Number of outputs to return

    Returns:
    --------
    None if return_outputs is 0 else a list of tuples containing
    (input, target, output)
    '''
    import numpy as np
    num_correct = 0
    num_pixels = 0
    coeffs = []
    losses = []
    outputs = []

    def should_append_outputs(p=0.5):
        '''
        Randomly decide whether to append outputs to the list
        Given that the number of outputs is less than return_outputs
        and a random number is less than probability p

        Parameters:
        ----------
        p: float, Default: 0.5
            Probability of returning True
        '''
        import random

        # clip p between 0 and 1
        p = min(1, max(0, p))
        return return_outputs > 0 and len(outputs) < return_outputs and p > random.random()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        if should_append_outputs():
            outputs.append((x, y, output))
        # compare y and output
        # L2 loss
        l2_loss = torch.nn.MSELoss()(output, y)
        # Correlation coefficient
        # convert to numpy
        output = output.cpu().numpy()
        coeff = np.corrcoef(output, y)
        if coeff >= 0.9:
            num_correct += 1
        coeffs.append(coeff)
        losses.append(l2_loss)

    print(f"Got {num_correct} / {len(loader)} correct with avg coeff {np.mean(coeffs)} and loss {np.mean(losses)}")



def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
