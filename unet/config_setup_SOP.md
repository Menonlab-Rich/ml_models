# U-Net Model Configuration Standard Operating Procedure (SOP)

This document outlines the standard procedure for configuring the U-Net machine learning model parameters, with a special focus on handling multi-channel images.

## System Parameters (Do Not Change)

- **DEVICE**: Automatically set based on CUDA availability. Ensure your setup supports CUDA for GPU acceleration.

## Adjustable Training Parameters

Adjust these parameters based on your dataset characteristics and training requirements:

- **LEARNING_RATE**: Starting learning rate for optimization. Begin with `2e-4` and adjust based on model performance and convergence.
- **BATCH_SIZE**: Number of samples per batch. Adjust according to your system's memory capacity.
- **NUM_EPOCHS**: Total number of epochs for training the model. Increase for more training or decrease for quicker iterations.
- **NUM_WORKERS**: Set this to the number of CPU cores available for efficient data loading.

## Handling Multi-Channel Images

- **CHANNELS_INPUT**: Set to the number of channels in your input images. Use `3` for RGB or `1` for grayscale images.
- **CHANNELS_OUTPUT**: Set according to the expected output image channels; `3` for RGB, `1` for grayscale.
- Note: Adjusting these parameters directly impacts how the model processes and generates images. Ensure compatibility with your data.

## Training Data Parameters

- **IMAGE_SIZE**: The target size for input images. Increase for higher resolution but consider memory limitations.
- **TRAIN_IMG_PATTERN** and **TARGET_IMG_PATTERN**: File path patterns for sourcing training and target images, respectively.

## Plotting Parameters

- **CMAP_IN** and **CMAP_OUT**: Colormaps for visualizing input and output images. Set based on the number of channels; typically `None` for multi-channel images to retain natural coloring.
- **PLOTTING_INTERPOLATION**: Set to `'nearest'` for single-channel and `'bilinear'` for multi-channel images to optimize visual clarity.
- **CBAR_MIN** and **CBAR_MAX**: Functions or values to define the color range for plotting. Important for standardized visual comparisons, especially in single-channel outputs.

## Model Saving and Loading Parameters

- **LOAD_MODEL**: Set `True` to continue training from a checkpoint; otherwise `False`.
- **SAVE_MODEL**: Set `True` to save the model post-training.
- **CHECKPOINT**: Filename for saving or loading model checkpoints.
- **EXAMPLES_DIR**: Directory for saving output visualizations and examples.

## Augmentation Pipelines

Adjust these to enhance model generalization and training effectiveness:

### Shared Transformations

For both input and target images:

- **transform_both**: Includes standard preprocessing like normalization and resizing. Extend with additional augmentations as needed, ensuring `ToTensorV2()` remains last.

### Input-Specific Transformations

For input images only:

- **transform_input**: Apply specific augmentations to simulate variations in input data (e.g., `A.GaussNoise(p=0.5)`). Avoid `ToTensorV2()` duplication.

### Target-Specific Transformations

For target images only:

- **transform_target**: Implement cautiously to maintain ground truth integrity. Typically less aggressive than input transformations.

## Procedure

1. Review and adjust training parameters considering dataset size, model complexity, and system capabilities.
2. Pay special attention to `CHANNELS_INPUT` and `CHANNELS_OUTPUT` for handling multi-channel images. Ensure data compatibility and correct preprocessing.
3. Configure data-related parameters to match your dataset's structure and requirements.
4. Set plotting parameters with a focus on the visual representation of multi-channel data. Ensure correct colormap and interpolation settings.
5. Review and adjust model saving/loading settings as per your training regimen.
6. Customize augmentation pipelines to balance between realistic variations and computational efficiency.

Ensure compliance with this SOP during model setup and training to maintain consistency and accuracy, particularly when dealing with multi-channel image data. Record all parameter changes for future reference and reproducibility.