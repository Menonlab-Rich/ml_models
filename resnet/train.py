from model import BCEResnet
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.loggers import NeptuneLogger
from os import environ
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from encoder.model import LitAutoencoder
from typing import List
from typing_extensions import override
from neptune.types import File


def load_encoder(ckpt_path: str):
    encoder = LitAutoencoder.load_from_checkpoint(ckpt_path, strict=False)
    return encoder


def main(config: Config, n_files: int = None):
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.data_dir, config.classes)
    test_loader = InputLoader(config.test_dir)
    test_target_loader = TargetLoader(config.test_dir, config.classes)

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
        pos_weight=list(config.weights.values()),
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


class Evaluator:
    class LitSkLearn:
        def __init__(self, model, **trainer_kwargs):
            self.model = model
            self.trainer = Trainer(**trainer_kwargs)

        def fit(self, X, y):
            self.trainer.fit(self.model, train_dataloaders=X, val_dataloaders=y)
            return self  # For chaining

        def predict(self, X):
            self.trainer.predict(self.model, dataloaders=X,
                                 return_predictions=True)

    class EvalLogger(DummyLogger):
        def __init__(self, save_metrics: List[str] = []):
            super().__init__()
            self.save_metrics = save_metrics
            self.accumulated_metrics = {}
            for metric in save_metrics:
                self.accumulated_metrics[metric] = 0.0

        @override
        def log_metrics(self, metrics, step):
            for metric_name, metric_value in metrics.items():
                if metric_name in self.save_metrics:
                    self.accumulated_metrics[metric_name] += metric_value
            print(f"Step {step}: {metrics}")

        def reset(self):
            for metric in self.save_metrics:
                self.accumulated_metrics[metric] = 0.0

    def __init__(self, neptune_logger: NeptuneLogger = None):
        self.neptune_logger = neptune_logger

    def kfold_cross_validation(
            self, model, input_loader, k=5, log_metrics: List[str] = []):
        logger = self.EvalLogger(save_metrics=log_metrics)

        model = BCEResnet(
            pos_weight=None,
            lr=config.learning_rate,
            n_channels=1,
        )
        trainer = Trainer(
            max_epochs=6,
            precision=config.precision,
            accelerator=config.accelerator,
            accumulate_grad_batches=2,  # Accumulate 2 batches before doing a backward pass
            logger=logger
        )
        results = []
        folds = input_loader.fold(k=k)
        for i, (train_ids, val_ids) in enumerate(folds):
            print(f"Fold {i + 1}")
            train_loader = InputLoader(config.data_dir, files=train_ids)
            target_loader = TargetLoader(config.data_dir, files=train_ids)
            data_module = ResnetDataModule(
                input_loader=train_loader, target_loader=target_loader,
                batch_size=config.batch_size, transforms=config.transform)
            trainer.fit(model, data_module)
            test_loader = InputLoader(config.data_dir, files=val_ids)
            test_target_loader = TargetLoader(config.data_dir, files=val_ids)
            test_data_module = ResnetDataModule(
                input_loader=test_loader, target_loader=test_target_loader,
                batch_size=config.batch_size, transforms=config.transform)
            trainer.test(model, datamodule=test_data_module)
            results.append(
                logger.accumulated_metrics["test_acc"] / config.epochs)
            logger.reset()

        return results

    def bias_variance_decomposition(
            self, model, input_loader, target_loader, n_repeats=5, k=5,
            log_metrics: List[str] = []):
        from mlxtend.evaluate import bias_variance_decomp
        train_inp_loader, val_inp_loader = input_loader.split(
            train_ratio=0.8, seed=16)
        train_tgt_loader, val_tgt_loader = target_loader.split(
            train_ratio=0.8, seed=16)
        train_data_module = ResnetDataModule(
            input_loader=train_inp_loader, target_loader=train_tgt_loader,
            batch_size=config.batch_size, transforms=config.transform)
        val_data_module = ResnetDataModule(
            input_loader=val_inp_loader, target_loader=val_tgt_loader,
            batch_size=config.batch_size, transforms=config.transform)

        estimator = self.LitSkLearn(
            model, logger=self.EvalLogger(save_metrics=log_metrics),
            max_epochs=6, precision=config.precision,
            accelerator=config.accelerator, accumulate_grad_batches=2)

        loss, bias, var = bias_variance_decomp(
            estimator, train_data_module.get_dataloader("train"),
            val_data_module.get_dataloader("val"),
            n_repeats=n_repeats, random_seed=16, fit_params={"model": model})
        log = f"Loss: {loss}, Bias: {bias}, Variance: {var}"
        self.neptune_logger.experiment["bias_variance_decomposition"] = log


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = Config(CONFIG_FILE_PATH)
    if args.debug:
        config.debug["enable"] = True
    main(config)
