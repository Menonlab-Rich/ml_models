from model import BCEResnet
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH
from pytorch_lightning.loggers import NeptuneLogger
from os import environ
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from encoder.model import LitAutoencoder


def load_encoder(ckpt_path: str):
    encoder = LitAutoencoder.load_from_checkpoint(ckpt_path, strict=False)
    return encoder


def main(config: Config, n_files: int = None):
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.data_dir)

    checkpoint_cb = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='resnet-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
        save_on_train_epoch_end=False,
    )

    model = BCEResnet(
        weight=None,
        lr=config.learning_rate,
        encoder=load_encoder(config.encoder_path),
        n_channels=config.input_channels,
    )

    logger = NeptuneLogger(
        api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
        project="richbai90/Resnet",  # format "workspace-name/project-name"
        tags=["training", "autoencoder", "resnet"],  # optional
    )

    debug = config.debug
    if debug['enable']:
        data_module = ResnetDataModule(
            input_loader, target_loader, batch_size=config.batch_size,
            transforms=config.transform,
            n_workers=1  # It takes time to spawn workers in debug mode so we set it to 1
        )
        Trainer(
            fast_dev_run=debug['fast'],
            limit_train_batches=debug['train_batches'],
            limit_val_batches=debug['val_batches'],
            callbacks=[checkpoint_cb]
        ).fit(model=model, datamodule=data_module)

    else:
        data_module = ResnetDataModule(
            input_loader, target_loader, batch_size=config.batch_size,
            transforms=config.transform)
        Trainer(
            logger=logger,
            max_epochs=config.epochs,
            precision=config.precision,
            accelerator=config.accelerator,
            callbacks=[checkpoint_cb]).fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    config = Config(CONFIG_FILE_PATH)
    main(config)
