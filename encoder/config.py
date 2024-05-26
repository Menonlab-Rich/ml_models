from base.config import BaseConfigHandler
from base.dataset import Transformer
from os import path as os_path
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Type, List
import torch

CONFIG_FILE_PATH = r"D:\CZI_scope\code\ml_models\encoder\config.yml"

# path custom tag handler
def path(loader, node):
    seq = loader.construct_sequence(node)
    return os_path.join(*seq)


# register the tag handlerpathjoin
yaml.add_constructor('!path', path)

def get_train_transform():
    pipeline = A.Compose([
        A.ToFloat(always_apply=True),
        ToTensorV2()
    ])
    
    return pipeline

def get_val_transform():
    pipeline = A.Compose([
        A.ToFloat(always_apply=True),
        ToTensorV2()
    ])
    
    return pipeline

class EncoderTransformer(Transformer):
    def __init__(self):
        super(EncoderTransformer, self).__init__(
            get_train_transform(),
            get_val_transform()
        )
        
    def apply_train(self, image):
        return self.train_transform(image=image)['image']
    
    def apply_val(self, image):
        return self.val_transform(image=image)['image']
    
    def __call__(self, inputs, targets) -> List[Type[torch.Tensor]]:
        inputs = self.apply_train(inputs)
        targets = self.apply_val(targets)
        return inputs, targets
    
class Config(BaseConfigHandler):
    def __init__(self, file_path: str):
        super(Config, self).__init__()
        self.config = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
        pipeline = A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2()
        ])
        
        def transform(x):
            return pipeline(image=x)['image']

        self.transform = {
            "train": EncoderTransformer(),
            "val": EncoderTransformer(),
        }

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

