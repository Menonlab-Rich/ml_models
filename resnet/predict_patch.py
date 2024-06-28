import numpy as np
import os
import tifffile as tiff
import argparse
from skimage.util import view_as_windows
from model import ResNet
from dataset import ResnetDataModule
from base.dataset import GenericDataLoader
from base.dataset import MockDataLoader
from pytorch_lightning import Trainer
from common import helpers, func_helpers

class PredictionLoader(GenericDataLoader):
    def __init__(self, patches):
        self.patches = func_helpers.flatten(patches)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], idx

    def get_ids(self, i=None):
        return range(len(self.patches)) if i is None else [i]

def parse_args():
    parser = helpers.DebugArgParser()
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the model checkpoint.", dest="model_path")
    parser.add_argument('--tiff-path', type=str,
                        help="Path to the combined TIFF file.")
    parser.add_argument(
        '--mask-path', type=str,
        help="Path to the corresponding mask file (npz).")
    parser.add_argument(
        '--patch-size', type=int, default=80,
        help="Size of the patches to extract.")
    return parser.parse_args()


def load_model(model_path: str, patches: list):
    prediction_loader = PredictionLoader(patches)

    model = ResNet.load_from_checkpoint(model_path, encoder=None, strict=False)
    data_module = ResnetDataModule.load_from_checkpoint(
        model_path, input_loader=MockDataLoader([]),
        target_loader=MockDataLoader([]),
        prediction_loader=prediction_loader,
        n_workers=4
    )
    return model, data_module


def load_combined_tiff_and_mask(tiff_path, mask_path):
    image_array = tiff.imread(tiff_path)
    mask_array = np.load(mask_path)['mask']
    return image_array, mask_array


def get_patches(image_array, mask_array, patch_size):
    patches = view_as_windows(
        image_array, (patch_size, patch_size),
        step=patch_size)
    mask_patches = view_as_windows(
        mask_array, (patch_size, patch_size),
        step=patch_size)
    return patches, mask_patches


def label_patch(mask_patch, threshold=0.5):
    total_pixels = mask_patch.size
    label_1_count = np.sum(mask_patch == 1)
    label_2_count = np.sum(mask_patch == 2)

    if label_1_count / total_pixels > threshold:
        return 1
    elif label_2_count / total_pixels > threshold:
        return 0
    else:
        return None


def process_patches(image_patches, mask_patches):
    labeled_patches = []

    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):
            mask_patch = mask_patches[i, j]
            if np.sum(
                    mask_patch != 0) < (
                    0.1 * mask_patch.size):  # Skip if the patch has predominantly zero values
                continue
            label = label_patch(mask_patch)
            if label is not None:
                image_patch = image_patches[i, j]
                labeled_patches.append((image_patch, label))

    return labeled_patches


def main():
    from tqdm import tqdm
    from common.func_helpers import flatten
    from PIL import Image
    args = parse_args()
    labeled_patches = []
    tiff_paths = os.listdir(args.tiff_path)
    mask_paths = os.listdir(args.mask_path)
    for tiff_path, mask_path in tqdm(zip(tiff_paths, mask_paths)):
        if not tiff_path.endswith('.tif') and not mask_path.endswith('.npz'):
            continue
        tiff_path = os.path.join(args.tiff_path, tiff_path)
        mask_path = os.path.join(args.mask_path, mask_path)
        image_array, mask_array = load_combined_tiff_and_mask(
            tiff_path, mask_path)
        image_patches, mask_patches = get_patches(
            image_array, mask_array, args.patch_size)
        labeled_patches.append(process_patches(
            image_patches, mask_patches))
    patches, labels = helpers.unzip(flatten(labeled_patches, 2))
    # Save the patches and labels as an image
    for i, (patch, label) in enumerate(zip(patches, labels)):
        patch = patch.squeeze()
        img = Image.fromarray(patch)
        img.save(fr"D:\CZI_scope\code\ml_models\resnet\examples\patch_{i}_label_{label}.tif")
    model_path = args.model_path
    model, data_module = load_model(model_path, patches)
    trainer = Trainer()
    output = trainer.predict(model, datamodule=data_module)
    output = [t.tolist() for t in output]
    output = flatten(output)
    confusion_matrix = np.zeros((2, 2))
    for i, label in tqdm(enumerate(labels)):
        if label == output[i]:
            confusion_matrix[label, label] += 1
        else:
            confusion_matrix[label, output[i]] += 1

    print(confusion_matrix)
    percentage_correct = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    print(f"Percentage correct: {percentage_correct}")


if __name__ == "__main__":
    main()
