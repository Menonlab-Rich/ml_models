from pytorch_lightning import Trainer
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config
from pytorch_lightning.loggers import NeptuneLogger
from os import environ


def load_models(resnet_ckpt_path):
    from model import ResNet
    model = ResNet.load_from_checkpoint(resnet_ckpt_path, encoder=None, strict=False)
    return model




def main(config: Config):
    
    logger = NeptuneLogger(
    api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
    project="richbai90/ResnetTest",  # format "workspace-name/project-name"
    tags=["testing", "resnet"],  # optional
)
    
    resnet_ckpt_path = r"checkpoints/resnet-epoch=05-accuracy_val=0.98.ckpt"
    model = load_models(resnet_ckpt_path)
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.data_dir, config.classes)
    data_module = ResnetDataModule(input_loader, target_loader, n_workers=5, transforms=config.transform, test_loaders='validation')
    model.eval()
    trainer = Trainer(logger=logger)
    trainer.test(model, datamodule=data_module)
    

if __name__ == '__main__':
    config = Config(r'config.yml')
    main(config)
