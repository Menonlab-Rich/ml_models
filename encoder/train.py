from model import WeightedLitAutoencoder, with_loss_fn, WeightedMSEMetric
from dataset import EncoderDataModule, InputLoader
from config import Config
from lightning.pytorch.loggers import NeptuneLogger
from os import environ
from pytorch_lightning import Trainer


def main(config: Config, n_files: int = None):
    input_loader = InputLoader(config.data_dir, n_files=n_files)

    data_module = EncoderDataModule(
        input_loader, batch_size=config.batch_size,
        transforms=config.transform)

    model = with_loss_fn(WeightedMSEMetric,
                         weights=list(config.weights.values()))(
        WeightedLitAutoencoder)(input_channels=config.input_channels,
                                embedding_dim=config.embedding_dim, size=config.resize,
                                lr=config.learning_rate, class_names=list(config.weights.keys()),)

    logger = NeptuneLogger(
        api_key=environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
        project="richbai90/AutoEncoder",  # format "workspace-name/project-name"
        tags=["training", "autoencoder"],  # optional
    )

    Trainer(max_epochs=config.epochs, logger=logger).fit(model, data_module)


if __name__ == '__main__':
    config = Config(r"D:\CZI_scope\code\ml_models\encoder\config.yml")
    main(config)
