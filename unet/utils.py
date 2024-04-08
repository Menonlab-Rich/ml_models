import config
import torch
import logging
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import Sequence
import torch.nn.functional as F


def generate_model_prediction(
        model, checkpoint_path, image_path, output_path, resize=1.):
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
    # optimizer is not needed for prediction
    load_checkpoint(model, None, checkpoint_path)
    # Load and preprocess the image
    image = Image.open(image_path)
    # check if resize is a float
    if isinstance(resize, float):
        image = image.resize(
            (int(image.width * resize),
             int(image.height * resize)))
    elif isinstance(resize, tuple):
        image = image.resize(resize)
    else:
        try:
            resize = float(resize)
            image = image.resize(
                (int(image.width * resize),
                 int(image.height * resize)))
        except:
            raise ValueError(
                "resize must be a float or a tuple of two integers")

    # Normalize and add batch dimension
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

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


def tensor_to_probability_map(tensor, classes: Sequence[int] = None):
    class_map = torch.tensor(classes, device=tensor.device)  # 3 classes corr
    probabilities = F.softmax(tensor, dim=0)
    predictions = torch.argmax(probabilities, dim=0)
    return class_map[predictions]


def evaluate_proability_model(model, loader, device=config.DEVICE):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_fake = model(x)
            y_pred = tensor_to_probability_map(y_fake)
            y_true = tensor_to_probability_map(y)
            # Calculate the accuracy
            accuracy = torch.sum(
                y_pred == y_true) / (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2])
            print(f"Accuracy: {accuracy.item()}")


def prepare_tensors_for_plotting(*img_tensors):
    # Convert to float
    np_imgs = []
    for img_tensor in img_tensors:
        img_tensor = img_tensor.to(dtype=torch.float32)

        # Calculate min and max values
        min_val = torch.min(img_tensor)
        max_val = torch.max(img_tensor)

        # Scale tensor to range 0 to 1
        scaled_tensor = (img_tensor - min_val) / (max_val - min_val)

        # Clamp values to ensure they are between 0 and 1
        scaled_tensor = torch.clamp(scaled_tensor, 0, 1)

        np_img = scaled_tensor.cpu().squeeze().numpy()
        if np_img.shape[0] == 3:
            np_img = np_img.transpose(1, 2, 0)

        np_imgs.append(np_img)

    return np_imgs if len(np_imgs) > 1 else np_imgs[0]


def get_color_range(img, min_fn, max_fn):
    '''
    Get the color range for an image

    Parameters:
    ----------
    img: torch.Tensor
        Image to get the color range for
    min_fn: function
        Function to calculate the minimum value for the color range
        If None, the minimum value of the image is used
        If a number, that number is used as the minimum value
    max_fn: function
        Function to calculate the maximum value for the color range
        If None, the maximum value of the image is used
        If a number, that number is used as the maximum value
    '''
    img = np.asarray(img)  # Convert to numpy array if not already
    if min_fn is None:
        vmin = img.min()
    elif callable(min_fn):
        vmin = min_fn(img)
    else:
        vmin = min_fn

    if max_fn is None:
        vmax = img.max()
    elif callable(max_fn):
        vmax = max_fn(img)
    else:
        vmax = max_fn

    return vmin, vmax


def logits_to_rgb(logits, color_map):
    """
    Converts logits from a model to an RGB image based on a provided color map.

    Parameters:
    - logits: torch.Tensor
        The raw output logits from the model. Shape: [N, C, H, W].
    - color_map: dict
        A dictionary mapping class indices to RGB colors.

    Returns:
    - torch.Tensor: An RGB image. Shape: [N, H, W, 3].
    """
    # Apply softmax to get probabilities and then argmax to get predicted class indices
    probs = torch.nn.functional.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)

    # Prepare an empty tensor for the RGB image
    rgb_image = torch.zeros(
        predictions.size(0),
        predictions.size(1),
        predictions.size(2),
        3, dtype=torch.uint8, device=logits.device)

    for class_index, color in color_map.items():
        mask = predictions == class_index
        for c in range(3):  # RGB channels
            rgb_image[mask, c] = color[c]

    # Moving the channel to the last dimension to match [N, H, W, 3] for plotting
    return rgb_image.permute(0, 2, 3, 1).cpu().numpy()


def map_to_rgb(y: np.ndarray, color_map: dict) -> np.ndarray:
    """
    Maps class indices to RGB colors based on a provided color map.

    Parameters:
    - y: np.ndarray
        The class indices. Shape: [N, H, W].
    - color_map: dict
        A dictionary mapping class indices to RGB colors.

    Returns:
    - np.ndarray: An RGB image. Shape: [N, H, W, 3].
    """
    # Prepare an empty tensor for the RGB image
    rgb_image = np.zeros(y.shape + (3,), dtype=np.uint8)
    for class_index, color in color_map.items():
        mask = y == class_index
        for c in range(3):  # RGB channels
            rgb_image[mask, c] = color[c]

    return rgb_image


def gen_evaluation_report(model, val_loader, device, task, multi_channel=False):
    '''
    Generate an evaluation report for the model
    Depending on the task, different metrics are calculated

    Parameters:
    ----------
    model: torch.nn.Module
        Model to evaluate
    val_loader: torch.utils.data.DataLoader
        DataLoader containing the validation data
    device: str
        Device to use for evaluation
    task: str
        Task to evaluate the model on. Either 'segmentation' or 'translation'
    multi_channel: bool, Default: False
        Whether the images have multiple channels (only relevant for translation task)
    '''
    if task == 'segmentation':
        from sklearn.metrics import jaccard_score, f1_score
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                # Assuming NCHW format and class dim=1
                preds = torch.argmax(logits, dim=1)

                # Move to CPU and convert to numpy for sklearn compatibility
                preds_np = preds.cpu().numpy().flatten()
                y_np = y.cpu().numpy().flatten()

                y_pred.extend(preds_np)
                y_true.extend(y_np)

            # Calculate metrics
            jaccard = jaccard_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            pixel_accuracy = np.mean(np.array(y_true) == np.array(y_pred))

            print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
            print(f"Jaccard Score: {jaccard:.4f}")
            print(f"F1 Score: {f1:.4f}")
    elif task == 'translation':
        from skimage.metrics import structural_similarity as ssim
        model.eval()
        losses = []
        ssim_scores = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_fake = model(x)
                loss = torch.nn.MSELoss()(y_fake, y)
                losses.append(loss.item())
                # Ensure correct conversion and handling of batches
                for i in range(y_fake.shape[0]):
                    ssim_score = ssim(
                        y_fake[i].cpu().numpy(),
                        y[i].cpu().numpy(),
                        multichannel=multi_channel)
                    ssim_scores.append(ssim_score)
        print(f"Mean Squared Error: {np.mean(losses):.4f}")
        print(f"Average SSIM: {np.mean(ssim_scores):.4f}")


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
        if config.TASK == 'segmentation':
            y_fake = logits_to_rgb(y_fake, config.COLOR_MAP)
            y = map_to_rgb(y, config.COLOR_MAP)
        # Prepare tensors for plotting
        y_fake = prepare_tensors_for_plotting(*y_fake)
        x = prepare_tensors_for_plotting(*x)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # Calculate dynamic range or use fixed values for color scaling
    vmin, vmax = get_color_range(y_fake, config.CBAR_MIN, config.CBAR_MAX)

    for i in range(6):
        row = i // 3
        col = i % 3

        axs[row, col].imshow(x[i], cmap=config.CMAP_IN, interpolation=config.PLOTTING_INTERPOLATION(
            config.CHANNELS_INPUT), vmin=vmin, vmax=vmax)
        axs[row, col].set_title(f"Input {i+1}")
        axs[row, col].axis('off')

        # Display predicted (RGB) image
        # Ensure y_fake is permuted from [C, H, W] to [H, W, C] for correct display
        axs[(row + 1) % 2, col].imshow(y_fake[i],
                                       cmap=config.CMAP_OUT, interpolation=config.PLOTTING_INTERPOLATION(
            config.CHANNELS_OUTPUT),
            vmin=vmin, vmax=vmax)
        axs[(row+1) % 2, col].set_title(f"Prediction {i+1}")
        axs[(row+1) % 2, col].axis('off')

        # Add a colorbar to the right of the figure
    if config.CBAR and config.CHANNELS_OUTPUT == 1:
        fig.subplots_adjust(right=0.85)  # Make room for the colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.05, 0.7])  # Position of colorbar
        # Normalization based on the vmin and vmax
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=config.CMAP_OUT, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

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


def split_dataset(dataset, split=0.8, save=False):
    '''
    Split the dataset into train and validation sets

    Parameters:
    ----------
    dataset: torch.utils.data.Dataset
        Dataset to split
    split: float, Default: 0.8
        Fraction of the dataset to use for training

    Returns:
    --------
    datasets: List[torch.utils.data.Dataset]
        The train and validation datasets in that order
    '''

    assert split > 0 and split < 1, "split must be in the range (0, 1)"

    sets = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * split), len(dataset) - int(len(dataset) * split)]
    )
    
    # save the datasets to a file
    if save:
        torch.save(sets[0], 'train_dataset.pth')
        torch.save(sets[1], 'val_dataset.pth')

    return sets

class LoggerOrDefault():
    '''
    LoggerOrDefault is a class that provides a logger object
    that can be used to log messages. If no logger is provided,
    a default logger is created and used.
    '''
    _logger = None

    def __init__(self) -> None:
        pass

    @classmethod
    def logger(cls, logger=None):
        '''
        Returns a logger object. If no logger is provided, a default
        logger is created and returned. Otherwise, the provided logger
        is returned. On the first call, the logger is created and stored
        in a class variable. Subsequent calls return the same logger.

        Parameters:
        ----------
        logger: logging.Logger, Default: None
            Logger to return. If None, a default logger is created and returned
            Or, if a logger was previously provided, the same logger is returned
        '''
        if logger is not None:
            cls._logger = logger
        if cls._logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            cls._logger = logger

        return cls._logger


if __name__ == "__main__":
    from model import UNet
    # Example usage
    model = UNet(in_channels=config.CHANNELS_INPUT,
                 out_channels=config.CHANNELS_OUTPUT).to(config.DEVICE)
    image_path = '/home/rich/Documents/school/menon/ml_models/unet/data/landscapes/gray/7128.jpg'
    output_path = './prediction.jpg'
    checkpoint_path = '/home/rich/Documents/school/menon/ml_models/unet/unet.pth.tar'
    generate_model_prediction(model, checkpoint_path,  image_path, output_path)
