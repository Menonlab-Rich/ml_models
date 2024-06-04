from pytorch_lightning import Trainer
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH
from pytorch_lightning.loggers import NeptuneLogger
from os import environ


def load_models(resnet_ckpt_path):
    from model import ResNet
    model = ResNet.load_from_checkpoint(resnet_ckpt_path, encoder=None, strict=False)
    data_module = ResnetDataModule.load_from_checkpoint(resnet_ckpt_path)
    return model, data_module




def main(config: Config):
    
    logger = NeptuneLogger(
    api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
    project="richbai90/ResnetTest",  # format "workspace-name/project-name"
    tags=["training", "autoencoder", "resnet"],  # optional
)
    
    resnet_ckpt_path = r"checkpoints/resnet-epoch=04-val_accuracy=0.98.ckpt"
    model, data_module = load_models(resnet_ckpt_path)
    
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
    config = Config(CONFIG_FILE_PATH)
    main(config)
