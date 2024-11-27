import torch.utils.data as data
import os
from os.path import join
from PIL import Image
from .utils import check_exceptions, is_image_file

class CityscapesDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        """
        Args:
            root_dir (string): Root directory of dataset where "leftImg8bit" and "gtFine" are located.
            split (string): One of "train", "val", or "test".
            transform (callable, optional): Optional transform to be applied on an image and target.
            preload_data (bool, optional): If True, all images and labels will be loaded into memory.
        """
        super(CityscapesDataset, self).__init__()

        # Set paths to image and label directories
        image_dir = join(root_dir, 'leftImg8bit', split)
        target_dir = join(root_dir, 'gtFine', split)

        self.image_filenames = []
        self.target_filenames = []

        # Walk through the city subdirectories and collect image and label file paths
        for city in [f for f in os.listdir(image_dir) if not f.startswith('.')]:
            city_image_dir = join(image_dir, city)
            city_target_dir = join(target_dir, city)

            # Get image and label file paths for the current city
            city_images = sorted([join(city_image_dir, f) for f in os.listdir(city_image_dir) if is_image_file(f)])
            city_labels = sorted([join(city_target_dir, f) for f in os.listdir(city_target_dir) if is_image_file(f)])

            # Ensure there's a one-to-one correspondence between images and labels in each city
            # assert len(city_images) == len(city_labels), f"Mismatch between images and labels in {city}."

            # Append the city files to the main lists
            self.image_filenames.extend(city_images)
            self.target_filenames.extend(city_labels)

        # Report the number of images in the dataset
        print(f'Number of {split} images: {len(self.image_filenames)}')

        self.transform = transform
        self.preload_data = preload_data

        if self.preload_data:
            # Preload data into memory (keep images in RGB format, labels as single-channel)
            print(f'Preloading {split} dataset...')
            self.raw_images = [Image.open(img_path).convert("RGB") for img_path in self.image_filenames]
            self.raw_labels = [Image.open(lbl_path) for lbl_path in self.target_filenames]
            print('Preloading complete.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a PIL image and label is a PIL image.
        """
        # Load image and label from memory or disk
        if not self.preload_data:
            input = Image.open(self.image_filenames[index]).convert("RGB")
            target = Image.open(self.target_filenames[index])
        else:
            input = self.raw_images[index]
            target = self.raw_labels[index]

        # Ensure image and label are consistent (check exceptions, e.g., dimensions match)
        check_exceptions(input, target)

        # Apply transformations if provided
        if self.transform:
            print(self.transform)
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
