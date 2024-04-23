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

def main():
    model = get_model(n_classes=NUM_CLASSES, n_channels=NUM_CHANNELS, multi_gpu=MULTI_GPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    load_checkpoint(model, optimizer, DEVICE, MODEL_PATH)
    data = load_data(os.path.join(DST_SAVE_DIR, 'val_set'))
    
    class ILoader(InputLoader):
        def __init__(self, data):
            self.files = sorted(data[0])
            self.directory = INPUT_PATH
    
    class ITLoader(TargetLoader):
        def __init__(self, data):
            files = sorted(data[0])
            self.classes = [1 if x[:3] == CLASS_MAPPING[1] else 0 for x in files]
    
    fp = r'D:\CZI_scope\code\preprocess_training\tifs\605-image_2024-04-03T18-01-17.77_0_page_14.tif'
    dataset = GenericDataset(input_loader=ILoader([[fp]]), target_loader=ITLoader([[fp]]), transform=TRANSFORMS)
    dataset.evaluate() # Set the dataset to evaluation mode
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    filenames = dataset.input_loader.get_ids()
    model.eval()
    ig = IntegratedGradients(model)
    
    for i, (input, target) in tqdm(enumerate(loader)):
        input, target = input.to(DEVICE), target.to(DEVICE)
        attribution = ig.attribute(input, target=target) # Compute the attribution
        input = input.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy() # Convert the input to a numpy array
        if len(input.shape) == 2:
            input = np.expand_dims(input, axis=-1) # Add a channel dimension if it doesn't exist
        
        
        target = target.squeeze().detach().cpu().numpy()
        attribution = attribution.squeeze().detach().cpu().numpy()
        if len(attribution.shape) == 2:
            attribution = np.expand_dims(attribution, axis=-1) # Add a channel dimension if it doesn't exist
        
        file_name = filenames[i]
        fig_hm, axs_hm = viz.visualize_image_attr(attribution, input, method='blended_heat_map', sign='all', show_colorbar=True)
        fig_cmp, axs_cmp = viz.visualize_image_attr_multiple(attribution, input, methods=['original_image', 'blended_heat_map'], signs=['all', 'all'], titles=['Original', 'Attribution'], fig_size=(12, 4))
        fig_overlay, axs_overlay = viz.visualize_image_attr(attribution, input, method='blended_heat_map', sign='all', show_colorbar=True, title='Overlayed Integrated Gradients')
        fig_hm.savefig(os.path.join(ROOT_DIR, '__test__interpretations', f'{file_name}_heatmap.png'))
        fig_cmp.savefig(os.path.join(ROOT_DIR, '__test__interpretations', f'{file_name}_comparison.png'))
        fig_overlay.savefig(os.path.join(ROOT_DIR, '__test__interpretations', f'{file_name}_overlay.png'))
    
    # display the visualizations in the web ui
    
        

if __name__ == '__main__':
    main()
        