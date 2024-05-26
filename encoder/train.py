from model import WeightedLitAutoencoder
from dataset import EncoderDataModule, InputLoader
from config import Config, CONFIG_FILE_PATH
from pytorch_lightning.loggers import NeptuneLogger
from os import environ
from pytorch_lightning import Trainer


def main(config: Config, n_files: int = None):
    input_loader = InputLoader(config.data_dir, n_files=n_files)

    data_module = EncoderDataModule(
        input_loader, batch_size=config.batch_size,
        transforms=config.transform)

    model = WeightedLitAutoencoder(input_channels=config.input_channels,
                                   embedding_dim=config.embedding_dim,
                                   size=config.resize, lr=config.learning_rate,
                                   weights=list(config.weights.values()),
                                   loss_scale=config.loss_scale,
                                   class_names=list(config.weights.keys()),)

    logger = NeptuneLogger(
        api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
        project="richbai90/AutoEncoder",  # format "workspace-name/project-name"
        tags=["training", "autoencoder"],  # optional
    )

    debug = config.debug
    if debug['enable']:
        Trainer(
            fast_dev_run=debug['fast'],
            limit_train_batches=debug['train_batches'],
            limit_val_batches=debug['val_batches'],
        ).fit(model, data_module)

    else:
        Trainer(
            logger=logger,
            max_epochs=config.epochs,
            precision=config.precision,
            accelerator=config.accelerator,
            progress_bar_refresh_rate=1,
        ).fit(model, data_module)


if __name__ == '__main__':
    config = Config(CONFIG_FILE_PATH)
    main(config)
