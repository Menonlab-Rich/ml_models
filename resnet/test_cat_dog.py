from model import BCEResnet
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH
from pytorch_lightning.loggers import NeptuneLogger
from os import environ, listdir, path
from pytorch_lightning import Trainer
from encoder.model import LitAutoencoder
from PIL import Image
import numpy as np

class CatDogInputLoader(InputLoader):
    def __init__(self, directory, n_files=None, files=None):
        super(CatDogInputLoader, self).__init__(directory, n_files, files)

class CatDogTargetLoader(TargetLoader):
    def __init__(self, directory, n_files=None, files=None):
        super(CatDogTargetLoader, self).__init__(directory, n_files, files)

    
    def _read(self, file):
        file = path.basename(file)
        if 'copy' in file: # If the file is a copy, it is a dog
            return 1
        
        return 0

def load_model(resnet_ckpt_path):
    from model import BCEResnet
    model = BCEResnet.load_from_checkpoint(resnet_ckpt_path, strict=False)
    
    return model

def main(config: Config, n_files: int = None):
    input_loader = CatDogInputLoader(r"D:\CZI_scope\code\ml_models\resnet\test_data\PetImages\combined\*.jpg")
    target_loader = CatDogTargetLoader(r"D:\CZI_scope\code\ml_models\resnet\test_data\PetImages\combined\*.jpg")

    model = load_model(r"D:\CZI_scope\code\.neptune\Untitled\RESNT-14\checkpoints\epoch=10-step=99.ckpt")
    model.eval()
    logger = NeptuneLogger(
        api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
        project="richbai90/ResnetTest",  # format "workspace-name/project-name"
        tags=["training", "autoencoder", "resnet"],  # optional
    )

    debug = {'enable': False, 'fast': True, 'train_batches': 1, 'val_batches': 1}
    if debug['enable']:
        data_module = ResnetDataModule(
            input_loader, target_loader, batch_size=10,
            transforms=config.transform,
            n_workers=1  # It takes time to spawn workers in debug mode so we set it to 1
        )
        Trainer(
            fast_dev_run=debug['fast'],
            limit_train_batches=debug['train_batches'],
            limit_val_batches=debug['val_batches'],
        ).fit(model=model, datamodule=data_module)

    else:
        data_module = ResnetDataModule(
            input_loader, target_loader, batch_size=10, no_split=True,
            transforms=config.transform)
        Trainer(
            logger=logger,
            max_epochs=config.epochs,
            precision=config.precision,
            accelerator=config.accelerator,
            log_every_n_steps=1, # Small dataset, so we log every step
        ).test(model=model, datamodule=data_module)


if __name__ == '__main__':
    config = Config(CONFIG_FILE_PATH)
    main(config)
