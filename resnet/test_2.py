from pytorch_lightning import Trainer
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config
from pytorch_lightning.loggers import NeptuneLogger
from tqdm import tqdm
from torch import sigmoid, tensor
import pandas as pd
from torch.nn import BCEWithLogitsLoss
from os import environ


def load_models(resnet_ckpt_path: str, config: Config):
    from model import BCEResNet
    model = BCEResNet.load_from_checkpoint(
        resnet_ckpt_path, encoder=None, strict=False)
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.data_dir, config.classes)
    data_module = ResnetDataModule.load_from_checkpoint(
        resnet_ckpt_path, test_loaders=(input_loader, target_loader))
    return model, data_module


def main(config: Config):

    resnet_ckpt_path = r"D:\CZI_scope\code\ml_models\resnet\checkpoints\resnet-epoch=04-val_accuracy=0.98.ckpt"
    model, data_module = load_models(resnet_ckpt_path, config)
    logger = NeptuneLogger(
        api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
        project="richbai90/ResnetTest",  # format "workspace-name/project-name"
        tags=["training", "autoencoder", "resnet"],  # optional
    )
    trainer_args = {
        "logger": logger,
        "max_epochs": config.epochs,
        "precision": config.precision,
        "accelerator": config.accelerator,
        "accumulate_grad_batches": 10,
    }

    trainer = Trainer(**trainer_args)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    config = Config(r'D:\CZI_scope\code\ml_models\resnet\config.yml')
    main(config)
