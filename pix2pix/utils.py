import torch

def save_checkpoint(state, path):
    torch.save(state, path)