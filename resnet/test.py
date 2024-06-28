from pytorch_lightning import Trainer
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH
from pytorch_lightning.loggers import NeptuneLogger
from os import environ


def load_models(resnet_ckpt_path):
    from model import ResNet
    input_loader = InputLoader(r"D:\CZI_scope\code\preprocess\superpixels")
    target_loader = TargetLoader(r"D:\CZI_scope\code\preprocess\superpixels", config.classes)
    model = ResNet.load_from_checkpoint(resnet_ckpt_path, encoder=None, strict=False)
    data_module = ResnetDataModule.load_from_checkpoint(resnet_ckpt_path, test_loaders=(input_loader, target_loader), n_workers=7)
    return model, data_module




def main(config: Config):
    
    logger = NeptuneLogger(
    api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
    project="richbai90/ResnetTest",  # format "workspace-name/project-name"
    tags=["training", "autoencoder", "resnet"],  # optional
)
    
    resnet_ckpt_path = r"D:\CZI_scope\code\ml_models\resnet\checkpoints\resnet-epoch=00-val_accuracy=0.93.ckpt"
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
