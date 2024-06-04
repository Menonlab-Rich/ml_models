import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import NeptuneLogger
from argparse import ArgumentParser
from model import BCEResnet
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH


def manual_validation(input, target, model, _loss, img_name):
    from PIL import Image
    from os import path
    import numpy as np
    import torch
    from torch.nn import BCEWithLogitsLoss
    
    config = Config(CONFIG_FILE_PATH)
    
    img_path = path.join(config.data_dir, img_name)
    img = np.array(Image.open(img_path))
    img = config.transform.apply_train(img)
    img = img.unsqueeze(0)
    
    expected_target = 0 if img_name[:3] == "605" else 1
    expected_target = config.transform.apply_train(expected_target, input=False).unsqueeze(0)
    
    assert target == expected_target, f"Target mismatch: {target} != {expected_target}"
    assert torch.allclose(input, img), f"Input mismatch: {input} != {img}"
    
    loss_fn = BCEWithLogitsLoss()
    pred = model(input)
    loss = loss_fn(pred, target)
    assert loss == _loss, f"Loss mismatch: {loss} != {_loss}"
    pred = model(img)
    loss = loss_fn(pred, expected_target)
    assert loss == _loss, f"Loss mismatch: {loss} != {_loss}"
    return loss
    


def main(config: Config, debug: bool = False, manual: bool = False):
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.data_dir, config.classes)

    data_module = ResnetDataModule(
        input_loader=input_loader,
        target_loader=target_loader,
        batch_size=1 if manual else config.batch_size,
        transforms=config.transform,
        n_workers=1 if debug else 4
    )

    logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/Resnet",
        tags=["training", "autoencoder", "resnet"]
    )

    checkpoint_cb = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='checkpoints',
        filename='resnet-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=3,
        mode='max',
        verbose=True
    )

    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    model = BCEResnet(
        lr=config.learning_rate,
        n_channels=config.input_channels,
    )

    trainer_args = {
        "logger": logger,
        "max_epochs": config.epochs,
        "precision": config.precision,
        "accelerator": config.accelerator,
        "accumulate_grad_batches": 10,
        "callbacks": [checkpoint_cb, swa]
    }

    if debug:
        trainer_args = {
            "fast_dev_run": True,
            "limit_train_batches": .1,
            "limit_val_batches": .01,
        }

    trainer = Trainer(**trainer_args)

    if manual:
        optimizer = model.configure_optimizers()
        data_module.setup('fit')
        for epoch in range(config.epochs):
            model.train()
            for i, (img, target, _) in enumerate(data_module.train_dataloader()):
                output = model(img)
                loss = model.loss_fn(output, target)
                model.backward(loss)
                model.optimizer_step(
                   epoch=epoch, batch_idx=i, optimizer=optimizer)
                logger.log_metrics({"train_loss": loss}, step=i)
                manual_validation(model=model, input=img, target=target, _loss=loss, img_name=_[0][0])
            model.eval()
            for i, (img, target, _) in enumerate(data_module.val_dataloader()):
                output = model(img)
                loss = model.loss_fn(output, target)
                logger.log_metrics({"val_loss": loss}, step=i)

            logger.log_metrics({"epoch": epoch}, step=epoch)
    else:
        trainer.fit(model, data_module)
        trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()

    config = Config(CONFIG_FILE_PATH)
    main(config, debug=args.debug, manual=args.manual)
