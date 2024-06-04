from model import ResNet
from dataset import ResnetDataModule, InputLoader
from base.dataset import MockDataLoader
from pytorch_lightning import Trainer
from common import helpers
import os


def parse_args():
    parser = helpers.DebugArgParser()
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the model checkpoint.", dest="model_path")
    parser.add_argument("image_path", type=str, nargs="+",
                        help="Path to the image to predict.")
    return parser.parse_args()


def load_model(model_path: str, image_paths: str):
    directories = [os.path.dirname(img_path) for img_path in image_paths]
    assert len(set(directories)) == 1, "All images must be in the same directory."
    directory = directories[0]
    prediction_loader = InputLoader(directory=directory, files=image_paths)
    model = ResNet.load_from_checkpoint(model_path, encoder=None, strict=False)
    data_module = ResnetDataModule.load_from_checkpoint(
        model_path, input_loader=MockDataLoader([]),
        target_loader=MockDataLoader([]),
        prediction_loader=prediction_loader,
        n_workers=1
    )
    return model, data_module


def main():
    from common.func_helpers import flatten
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path
    model, data_module = load_model(model_path, image_path)
    trainer = Trainer()
    output = trainer.predict(model, datamodule=data_module)
    output = [t.tolist() for t in output]
    output = flatten(output)
    print(output)


if __name__ == '__main__':
    main()
