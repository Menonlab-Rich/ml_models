from pytorch_lightning import Trainer
from model import BCEResnet
from dataset import GenericDataModule



def main():
    data_module = GenericDataModule()
    model = BCEResnet()
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, datamodule=data_module)