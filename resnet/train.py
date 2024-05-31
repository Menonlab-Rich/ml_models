from model import BCEResnet
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH
from pytorch_lightning.loggers import NeptuneLogger
from os import environ
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from encoder.model import LitAutoencoder

from pytorch_lightning.callbacks import Callback


class TestAfterValidationCallback(Callback):
    def __init__(self, datamodule, debug=False):
        super().__init__()
        self.datamodule = datamodule
        self.debug = debug

    def on_validation_epoch_end(self, trainer, pl_module):
        # Run the test loop
        if not trainer.running_sanity_check and not self.debug:
            trainer.test(datamodule=self.datamodule)


def load_encoder(ckpt_path: str):
    encoder = LitAutoencoder.load_from_checkpoint(ckpt_path, strict=False)
    return encoder


def main(config: Config, n_files: int = None):
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.data_dir)
    test_loader = InputLoader(config.test_dir)
    test_target_loader = TargetLoader(config.test_dir)

    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    checkpoint_cb = ModelCheckpoint(
        monitor='test_accuracy',
        dirpath='checkpoints',
        filename='resnet-{epoch:02d}-{test_accuracy:.2f}',
        save_top_k=3,
        mode='max',
        save_on_train_epoch_end=False,
        verbose=True
    )

    model = BCEResnet(
        weight=None,
        lr=config.learning_rate,
        n_channels=1,
    )

    logger = NeptuneLogger(
        api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
        project="richbai90/Resnet",  # format "workspace-name/project-name"
        tags=["training", "autoencoder", "resnet"],  # optional
    )

    debug = config.debug
    if debug['enable']:
        data_module = ResnetDataModule(
            input_loader=input_loader, target_loader=target_loader, test_loaders=(
                test_loader, test_target_loader), batch_size=config.batch_size,
            transforms=config.transform,
            n_workers=1  # It takes time to spawn workers in debug mode so we set it to 1
        )
        testing_cb = TestAfterValidationCallback(
            datamodule=data_module, debug=True)
        Trainer(
            fast_dev_run=debug['fast'],
            limit_train_batches=debug['train_batches'],
            limit_val_batches=debug['val_batches'],
            callbacks=[checkpoint_cb, testing_cb, swa]
        ).fit(model=model, datamodule=data_module)

    else:
        data_module = ResnetDataModule(
            input_loader=input_loader, target_loader=target_loader,
            batch_size=config.batch_size, transforms=config.transform)
        testing_cb = TestAfterValidationCallback(datamodule=data_module)
        Trainer(
            logger=logger,
            max_epochs=config.epochs,
            precision=config.precision,
            accelerator=config.accelerator,
            accumulate_grad_batches=3,  # Accumulate 3 batches before doing a backward pass
            callbacks=[checkpoint_cb, testing_cb, swa]).fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    config = Config(CONFIG_FILE_PATH)
    main(config)
