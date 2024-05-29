from pytorch_lightning import Trainer
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config
from pytorch_lightning.loggers import NeptuneLogger
from os import environ


def load_models(resnet_ckpt_path, encoder_ckpt_path):
    from model import ResNet
    from encoder.model import LitAutoencoder
    encoder = LitAutoencoder.load_from_checkpoint(encoder_ckpt_path, strict=False)
    encoder.eval() # Freeze the encoder (should be done already in the model but just in case)
    model = ResNet.load_from_checkpoint(resnet_ckpt_path, encoder=encoder, strict=False)
    return model




def main(config: Config):
    
    logger = NeptuneLogger(
    api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
    project="richbai90/ResnetTest",  # format "workspace-name/project-name"
    tags=["training", "autoencoder", "resnet"],  # optional
)

    
    resnet_ckpt_path = r"D:\CZI_scope\code\ml_models\resnet\best_model.ckpt"
    encoder_ckpt_path = r"D:\CZI_scope\code\ml_models\encoder\best_model.ckpt"
    model = load_models(resnet_ckpt_path, encoder_ckpt_path)
    input_loader = InputLoader(r'D:\CZI_scope\code\preprocess\validation')
    target_loader = TargetLoader(r'D:\CZI_scope\code\preprocess\validation')
    data_module = ResnetDataModule(
        input_loader, target_loader, batch_size=8,
        transforms=config.transform, no_split=True
    )
    
    model.eval()
    model.freeze()
    trainer = Trainer(logger=logger)
    trainer.test(model, datamodule=data_module)
    

if __name__ == '__main__':
    config = Config(r'D:\CZI_scope\code\ml_models\resnet\config.yml')
    main(config)