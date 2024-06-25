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

def ToTensorLong(x: np.ndarray) -> torch.Tensor:
    '''
    Convert a number to a tensor of type long
    '''
    x = np.array(x, dtype=np.uint8)
    return torch.tensor(x, dtype=torch.long)


def get_train_transform():
    pca = PCA(n_components=3)
    return {
        "input": A.Compose([
            A.ToFloat(always_apply=True),
            A.LongestMaxSize(max_size=512),
            ToTensorV2(),
            
        ]),
        "target": ToTensorLong
    }


def get_val_transform():
    return {
        "input": A.Compose([
            A.ToFloat(always_apply=True),
            A.LongestMaxSize(max_size=512),
            ToTensorV2(),
        ]),
        "target": ToTensorLong
    }


class UnetTransformer(Transformer):
    def __init__(self):
        super(UnetTransformer, self).__init__(
            get_train_transform(),
            get_val_transform()
        )

    def apply_train(self, input=True, **kwargs):
        if input:
            xformed = self.train_transform['input'](image=kwargs.get('image'), mask=kwargs.get('mask'))
            return xformed['image'], xformed['mask']
        else:
            return self.train_transform['target'](kwargs.get('mask'))

    def apply_val(self, input=True, **kwargs):
        if input:
            xformed = self.train_transform['input'](image=kwargs.get('image'), mask=kwargs.get('mask'))
            return xformed['image'], xformed['mask']
        else:
            return self.train_transform['target'](kwargs.get('mask'))

    def __call__(self, inputs, targets) -> List[Type[torch.Tensor]]:
        inputs, targets = self.apply_train(input=True, image=inputs, mask=targets)
        targets = self.apply_val(input=False, mask=targets)
        return inputs, targets


class Config(BaseConfigHandler):
    def __init__(self, file_path: str):
        super(Config, self).__init__()
        self.config = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
        pipeline = A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2()
        ])

        self.transform = UnetTransformer()

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
