# UNet Model Implementation

This repository contains an implementation of the UNet model, a popular architecture for semantic segmentation tasks. The UNet model is designed to work with images, providing a mechanism for effectively segmenting an image into its constituent parts or classes. This implementation is structured to be modular, easy to understand, and customizable for various segmentation tasks.

## Repository Structure

- [`config.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/config.py): Contains configuration settings for the model, including device setup, learning rate, batch size, image size, input/output channels, loss function, and paths for training images, target images, and where to save results.
- [`dataset.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/dataset.py): Defines a custom PyTorch dataset class for loading and transforming images for training and validation.
- [`model.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/model.py): Implements the UNet architecture using PyTorch, including the contracting path (encoder), expansive path (decoder), and the final convolution layer.
- [`requirements.txt`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/requirements.txt): Lists all the necessary Python packages needed to run the model.
- [`train.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/train.py): Contains the training loop for the UNet model, including loading the dataset, setting up the model, optimizer, and loss function, and saving the model after training.
- [`utils.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/utils.py): Provides utility functions for model checkpointing and generating sample predictions during training.

## Features

- **Customizable Model Configuration**: Easily adjust model parameters such as learning rate, batch size, and image size through the `config.py` file.
- **Dataset Flexibility**: The `dataset.py` script allows for easy adaptation to different datasets by modifying the input and target image paths.
- **Modular UNet Implementation**: The `model.py` file provides a clear and modular implementation of the UNet architecture, making it easy to understand and modify.
- **Training and Validation**: The `train.py` script facilitates the training process, including data loading, model training, validation, and saving checkpoints.
- **Utility Functions**: Additional utility functions in `utils.py` support checkpoint saving and loading, as well as generating output examples for evaluation.

## Advanced Image Augmentations with Albumentations

This UNet implementation utilizes the Albumentations library, a fast and flexible tool for image augmentations, which is crucial for improving model generalization on image segmentation tasks. Albumentations provides a wide range of augmentation techniques that can be easily integrated into your data preprocessing pipeline.

### Setting Up Albumentations

The `config.py` file includes a section dedicated to defining the augmentation pipeline for both input and target images. This setup ensures that the model is exposed to a variety of image conditions, thereby enhancing its ability to generalize across unseen data.

Here's a brief overview of how to set up and customize your augmentation pipeline using Albumentations within this project:

1. **Install Albumentations**: Ensure that Albumentations is installed. It is listed in the `requirements.txt`, so running `pip install -r requirements.txt` will take care of this step.

2. **Define Augmentations in `config.py`**: The augmentation pipeline is defined using the Albumentations library. Two separate compositions are provided by default:
   - `transform_input`: Augmentations applied only to the input images. For example, adding Gaussian noise to simulate real-world imperfections.
   - `transform_both`: Augmentations applied to both the input and target images simultaneously. This ensures that any geometric transformations maintain spatial consistency between the input and target.

A third augmentation pipeline, `transform_target`, can be defined if you need to apply augmentations to the target images only.

3. **Customize Your Augmentations**: You can customize your augmentation pipeline by adding or modifying the transformations in `config.py`. Albumentations provides a wide range of options, including but not limited to rotations, flips, scaling, cropping, and various kinds of noise. Documentation and a full list of available transformations can be found on the [Albumentations GitHub page](https://github.com/albumentations-team/albumentations). You can also create custom transformations using the `Lambda` transformation.

### Example Augmentation Setup

Here is an example snippet from `config.py` showing how to set up a simple augmentation pipeline:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentation pipeline for both input and target images
transform_both = A.Compose([
    # Example: Resize images to the specified size
    # A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
    # Normalize pixel values and convert images to tensors
    ToTensorV2(),
], additional_targets={'target': 'image'})

# Augmentation pipeline for input images only
transform_input = A.Compose([
    # Example: Add Gaussian noise to input images
    A.GaussNoise(p=0.5),
])
```

### Integrating Augmentations into the Dataset

The custom dataset class `Dataset` in `dataset.py` is designed to apply these transformations during the data loading process. When initializing the `Dataset` object, you can pass the defined transformations (`transform_input`, `transform_both`) as arguments. The dataset class handles the application of these transformations to ensure that the model receives augmented data during training.

By leveraging Albumentations, you can significantly enhance the diversity of your training dataset, which is key to training robust deep learning models for image segmentation tasks like those performed by the UNet model in this repository.

## Getting Started

To get started with this UNet implementation, first clone the repository and install the required dependencies:

```bash
git clone https://github.com/Menonlab-Rich/ml_models.git
cd ml_models/unet
pip install -r requirements.txt
```

Adjust the configuration settings in `config.py` as needed for your specific dataset and hardware setup. Then, you can train the model by running:

```bash
python train.py
```

For more detailed instructions and customization options, please refer to the comments within each script.