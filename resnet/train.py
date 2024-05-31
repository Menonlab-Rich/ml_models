from model import BCEResnet
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH
from pytorch_lightning.loggers import NeptuneLogger
from os import environ
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from encoder.model import LitAutoencoder


def load_encoder(ckpt_path: str):
    encoder = LitAutoencoder.load_from_checkpoint(ckpt_path, strict=False)
    return encoder


def main(config: Config, n_files: int = None):
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.data_dir)
    test_loader = InputLoader(config.test_dir)
    test_target_loader = TargetLoader(config.test_dir)

    test_data_module = ResnetDataModule(
        input_loader=input_loader, target_loader=target_loader,
        batch_size=config.batch_size, transforms=config.transform)

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

        Trainer(
            fast_dev_run=debug['fast'],
            limit_train_batches=debug['train_batches'],
            limit_val_batches=debug['val_batches'],
            callbacks=[checkpoint_cb, swa]
        ).fit(model=model, datamodule=data_module)

    else:
        data_module = ResnetDataModule(
            input_loader=input_loader, target_loader=target_loader,
            batch_size=config.batch_size, transforms=config.transform)
        trainer = Trainer(
            logger=logger,
            max_epochs=config.epochs,
            precision=config.precision,
            accelerator=config.accelerator,
            accumulate_grad_batches=3,  # Accumulate 3 batches before doing a backward pass
            callbacks=[checkpoint_cb, swa])

        trainer.fit(model, data_module)
        trainer.test(model, datamodule=test_data_module)


def evaluate(config: Config):
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.data_dir)
    folds = input_loader.fold(k=config.k_folds)
    # TODO: Implement k-fold cross-validation


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = Config(CONFIG_FILE_PATH)
    if args.debug:
        config.debug["enable"] = True
    main(config)
