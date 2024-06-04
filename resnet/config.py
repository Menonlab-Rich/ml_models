from base.config import BaseConfigHandler
from base.dataset import Transformer
from os import path as os_path
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Type, List, Union
import torch
import numpy as np
from sklearn.decomposition import PCA

CONFIG_FILE_PATH = r'D:\CZI_scope\code\ml_models\resnet\config.yml' 

# path custom tag handler


def path(loader, node):
    seq = loader.construct_sequence(node)
    return os_path.join(*seq)


# register the tag handlerpathjoin
yaml.add_constructor('!path', path)


def ToTensorFloat(x: np.ndarray) -> torch.Tensor:
    '''
    Convert a number to a tensor of type long
    '''
    x = np.array(x, dtype=np.uint8)
    return torch.tensor(x, dtype=torch.float)


def get_train_transform():
    pca = PCA(n_components=3)
    return {
        "input": A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2()
        ]),
        "target": ToTensorFloat
    }


def get_val_transform():
    return {
        "input": A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2()
        ]),
        "target": ToTensorFloat
    }


class ResnetTransformer(Transformer):
    def __init__(self):
        super(ResnetTransformer, self).__init__(
            get_train_transform(),
            get_val_transform()
        )

    def apply_train(self, x, input=True):
        if input:
            return self.train_transform['input'](image=x)['image']
        else:
            return self.train_transform['target'](x)

    def apply_val(self, x, input=True):
        if input:
            return self.val_transform['input'](image=x)['image']
        else:
            return self.val_transform['target'](x)

    def __call__(self, inputs, targets) -> List[Type[torch.Tensor]]:
        inputs = self.apply_train(inputs, input=True)
        targets = self.apply_val(targets, input=False)
        return inputs, targets


class Config(BaseConfigHandler):
    def __init__(self, file_path: str):
        super(Config, self).__init__()
        self.config = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
        pipeline = A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2()
        ])

        self.transform = ResnetTransformer()

    def __getattr__(self, name: str) -> any:
        return self.config[name]

    def get(self, key):
        return self.config[key]

    def set(self, key, value):
        self.config[key] = value

    def update(self, key, value):
        self.set(key, value)

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    def load(self, path):
        with open(path, 'r') as f:
            for line in f:
                k, v = line.strip().split('=')
                self.set(k, v)

    def __str__(self):
        return str(self.config)
