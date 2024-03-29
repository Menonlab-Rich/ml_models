import os
import shutil
import argparse
from random import shuffle

'''
A simple script to split a dataset into training and validation sets.
Assumes that the dataset consists of image files and corresponding annotation files.
Images and annotations are expected to have the same name, with different extensions.
'''

def split_dataset(images_folder, annotations_folder, ratio):
    # Ensure input paths are absolute
    images_folder = os.path.abspath(images_folder)
    annotations_folder = os.path.abspath(annotations_folder)

    # Create directories for the split datasets if they don't exist
    train_images_folder = os.path.join(images_folder, 'train')
    train_annotations_folder = os.path.join(annotations_folder, 'train')
    val_images_folder = os.path.join(images_folder, 'val')
    val_annotations_folder = os.path.join(annotations_folder, 'val')

    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_annotations_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_annotations_folder, exist_ok=True)

    # List all image files
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    shuffle(image_files)  # Shuffle to randomize distribution

    # Calculate the split index
    split_index = int(len(image_files) * ratio)

    # Split into training and validation sets
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Function to copy files to their new location
    def move_files(files, src_folder, dst_folder):
        for file in files:
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(dst_folder, file)
            shutil.move(src_path, dst_path)

    # Move image and annotation files
    move_files(train_files, images_folder, train_images_folder)
    move_files(val_files, images_folder, val_images_folder)

    # Move corresponding annotation files
    for file in train_files + val_files:
        annotation_file = os.path.splitext(file)[0] + '.xml'
        src_annotation_path = os.path.join(annotations_folder, annotation_file)
        if file in train_files:
            dst_annotation_path = os.path.join(train_annotations_folder, annotation_file)
        else:
            dst_annotation_path = os.path.join(val_annotations_folder, annotation_file)
        if os.path.exists(src_annotation_path):
            shutil.move(src_annotation_path, dst_annotation_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a dataset into training and validation sets.")
    parser.add_argument("images_folder", type=str, help="Path to the folder containing image files.")
    parser.add_argument("annotations_folder", type=str, help="Path to the folder containing annotation files.")
    parser.add_argument("ratio", type=float, help="Ratio of the dataset to be used for training (between 0 and 1).")

    args = parser.parse_args()

    if not 0 <= args.ratio <= 1:
        raise ValueError("The ratio must be between 0 and 1.")

    split_dataset(args.images_folder, args.annotations_folder, args.ratio)
