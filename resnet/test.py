from pytorch_lightning import Trainer
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config
from pytorch_lightning.loggers import NeptuneLogger
from os import environ


def load_models(resnet_ckpt_path):
    from model import ResNet
    from encoder.model import LitAutoencoder
    model = ResNet.load_from_checkpoint(resnet_ckpt_path, encoder=None, strict=False)
    return model




def main(config: Config):
    
    logger = NeptuneLogger(
    api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
    project="richbai90/ResnetTest",  # format "workspace-name/project-name"
    tags=["training", "autoencoder", "resnet"],  # optional
)
    
    resnet_ckpt_path = r"D:\CZI_scope\code\ml_models\resnet\checkpoints\resnet-epoch=03-accuracy_val=0.97.ckpt"
    model = load_models(resnet_ckpt_path)
    data_module = ResnetDataModule(
        input_loader=InputLoader(config.data_dir), target_loader=TargetLoader(config.data_dir, config.classes),
        batch_size=config.batch_size, transforms=config.transform, test_loaders="validation"
    )
    
    
    trainer = Trainer(logger=logger)
    trainer.test(model, datamodule=data_module)
    

if __name__ == '__main__':
    config = Config(r'D:\CZI_scope\code\ml_models\resnet\config.yml')
    main(config)