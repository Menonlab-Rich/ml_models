# ML Models Repository Overview

This repository is a collection of machine learning models with a focus on deep learning architectures for image processing tasks. It includes implementations of two significant models: UNet and Pix2Pix. These models are widely used for tasks such as image segmentation (UNet) and image-to-image translation (Pix2Pix).

## Repository Structure

The repository is organized into two main directories, each dedicated to a specific model:

- [`UNet`](https://github.com/Menonlab-Rich/ml_models/tree/main/unet): Contains the implementation of the UNet model, designed for efficient image segmentation tasks. The directory includes Python scripts for the model architecture, dataset preparation, training routines, and utility functions. A detailed `readme.md` provides instructions and insights into the UNet model's implementation and usage.

- [`Pix2Pix`](https://github.com/Menonlab-Rich/ml_models/tree/main/pix2pix): Hosts the implementation of the Pix2Pix model, a popular framework for image-to-image translation tasks using conditional GANs. This section includes scripts for the generator and discriminator networks, configuration settings, dataset handling, and the training process.

### UNet Directory

- [`config.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/config.py): Configuration settings for the UNet model training and evaluation.
- [`dataset.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/dataset.py): Dataset preparation and loading utilities for UNet.
- [`model.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/model.py): The UNet model architecture.
- [`requirements.txt`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/requirements.txt): Lists dependencies for running the UNet model.
- [`train.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/train.py): Script to train the UNet model.
- [`utils.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/utils.py): Utility functions supporting the UNet implementation.

### Pix2Pix Directory

- [`config.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/pix2pix/config.py): Configuration settings for the Pix2Pix model training and evaluation.
- [`dataset.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/pix2pix/dataset.py): Handles dataset loading and preprocessing for Pix2Pix.
- [`discriminator.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/pix2pix/discriminator.py): The discriminator network for the Pix2Pix model.
- [`generator.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/pix2pix/generator.py): The generator network for the Pix2Pix model.
- [`requirements.txt`](https://github.com/Menonlab-Rich/ml_models/blob/main/pix2pix/requirements.txt): Lists dependencies for running the 
- [`train.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/pix2pix/train.py): Script to train the Pix2Pix model.
- [`utils.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/pix2pix/utils.py): Utility functions for the Pix2Pix implementation.

## Getting Started

To get started with either the UNet or Pix2Pix models, clone the repository and navigate to the respective directory. Install the required dependencies listed in the `requirements.txt` file of the chosen model directory. Follow the instructions provided in the `readme.md` file within the UNet directory for detailed setup and training procedures. For Pix2Pix, refer to the comments within each script for guidance on configuration, dataset preparation, and training. The readme file for Pix2Pix is in progress and will be available soon.

This repository offers a solid foundation for researchers and practitioners looking to explore or apply UNet and Pix2Pix models to their image processing tasks.

## A note on Pix2Pix

The pix2pix model is much less generic than the UNet model, and is coded with a specific use case in mind. It therefore will require
more modification to be used for other tasks. The UNet model is more generic and can be used for a wider variety of tasks. In the future I hope to make this model more generic as well.