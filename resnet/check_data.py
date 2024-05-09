import torch
from torch.utils.data import DataLoader
from dataset import GenericDataset
from utils import load_data
from config import CLASS_MAPPING

def main(filepath):
    inputs, targets = load_data(filepath)
    for i, t in zip(inputs, targets):
        if i[:3] == CLASS_MAPPING[1]:
            assert t == 1, f'Expected target to be 1, but got {t}'
        elif i[:3] == CLASS_MAPPING[0]:
            assert t == 0, f'Expected target to be 0, but got {t}'
        else:
            raise ValueError(f'Invalid class found in input: {i}')
    
    print('Data is consistent')
    
if __name__ == '__main__':
    main(r'D:\CZI_scope\code\ml_models\resnet\val_set')