from captum.attr import IntegratedGradients, visualization as viz
from model import get_model
from dataset import GenericDataset, GenericDataLoader
from config import InputLoader, TargetLoader, MODEL_PATH, NUM_CLASSES, NUM_CHANNELS, \
    MULTI_GPU, LEARNING_RATE, CLASS_MAPPING, TRANSFORMS, DEVICE, DST_SAVE_DIR, INPUT_PATH, \
    ROOT_DIR
from utils import load_checkpoint, load_data
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def main():
    model = get_model(n_classes=NUM_CLASSES,
                      n_channels=NUM_CHANNELS, multi_gpu=MULTI_GPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    load_checkpoint(model, optimizer, DEVICE, MODEL_PATH)
    data = load_data(os.path.join(DST_SAVE_DIR, 'val_set'))

    class ILoader(InputLoader):
        def __init__(self, data):
            files_605 = [x for x in data if x[:3] == CLASS_MAPPING[0]]
            files_625 = [x for x in data if x[:3] == CLASS_MAPPING[1]]
            files_605 = np.random.choice(files_605, 5)
            files_625 = np.random.choice(files_625, 5)
            self.files = np.concatenate([files_605, files_625])
            self.directory = INPUT_PATH

    class ITLoader(TargetLoader):
        def __init__(self, data):
            files = sorted(data)
            self.classes = [1 if x[:3] ==
                            CLASS_MAPPING[1] else 0 for x in files]
    input_loader = ILoader(data[0])
    # Use the same files as the input loader
    target_loader = ITLoader(input_loader.get_ids())
    dataset = GenericDataset(
        input_loader=input_loader, target_loader=target_loader,
        transform=TRANSFORMS)
    dataset.eval()  # Set the dataset to evaluation mode
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    filenames = dataset.input_loader.get_ids()
    model.eval()
    ig = IntegratedGradients(model)

    for i, (input, target) in tqdm(enumerate(loader)):
        input, target = input.to(DEVICE), target.to(DEVICE)
        # Compute the attribution
        attribution = ig.attribute(input, target=target)
        input = input.permute(0, 2, 3, 1).squeeze().detach(
        ).cpu().numpy()  # Convert input to numpy array
        if len(input.shape) == 2:
            # Add a channel dimension if it doesn't exist
            input = np.expand_dims(input, axis=-1)

        attribution = attribution.squeeze().detach().cpu().numpy()
        if len(attribution.shape) == 2:
            # Add a channel dimension if it doesn't exist
            attribution = np.expand_dims(attribution, axis=-1)

        file_name = filenames[i]
        # Generate Captum visualizations
        # fig_hm, axs_hm = viz.visualize_image_attr(attribution, input, method='blended_heat_map', sign='all', show_colorbar=True)
        fig_cmp, axs_cmp = viz.visualize_image_attr_multiple(
            attribution, input, methods=['original_image', 'heat_map'],
            signs=['all', 'positive'],
            titles=['Original', 'Attribution'],
            fig_size=(15, 8),
            show_colorbar=True, cmap='viridis')
        # fig_overlay, axs_overlay = viz.visualize_image_attr(attribution, input, method='blended_heat_map', sign='all', show_colorbar=True, title='Overlayed Integrated Gradients')

        # fig_hm.suptitle(f'{file_name} - Heatmap\nPrediction: {CLASS_MAPPING[int(target)]}')
        fig_cmp.suptitle(
            f'{file_name} - Comparison\nPrediction: {CLASS_MAPPING[int(target)]}\n\n')
        # fig_overlay.suptitle(f'{file_name} - Overlay\nPrediction: {CLASS_MAPPING[int(target)]}')

        fig_cmp.savefig(os.path.join(
            ROOT_DIR, f'{file_name}_comparison_positive.png'))
        # fig_hm.savefig(os.path.join(ROOT_DIR, f'{file_name}_heatmap.png'))
        # fig_overlay.savefig(os.path.join(ROOT_DIR, f'{file_name}_overlay.png'))


if __name__ == '__main__':
    main()
