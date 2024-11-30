import os
import nibabel as nib
import numpy as np
import re

def load_nifti(filepath):
    """Load a .nii file."""
    nii_img = nib.load(filepath)
    return nii_img.get_fdata(), nii_img.affine

def save_nifti(data, affine, output_path):
    """Save a NumPy array as a .nii file."""
    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, output_path)
    print(f"Saved processed file to {output_path}")

def adapt_image_axes(image_path, label_path, output_path):
    """
    Adapt the axes of the image to match the label's shape.

    Parameters:
        image_path (str): Path to the image .nii file.
        label_path (str): Path to the label .nii file.
        output_path (str): Path to save the adapted image .nii file.
    """
    # Load image and label
    image_data, image_affine = load_nifti(image_path)
    label_data, _ = load_nifti(label_path)

    # Check and adapt the image shape
    if image_data.shape != label_data.shape:
        # Find the permutation to match the label's shape
        for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            if image_data.transpose(perm).shape == label_data.shape:
                print(f"Adapting image axes {image_data.shape} -> {label_data.shape} using permutation {perm}")
                image_data = image_data.transpose(perm)
                break
        else:
            raise ValueError(f"Cannot adapt image shape {image_data.shape} to label shape {label_data.shape}")

    # Save the adapted image
    save_nifti(image_data, image_affine, output_path)

def get_numeric_id(filename, prefix, suffix):
    """
    Extract the numeric ID from a filename.
    Example: 'image00001.nii' -> 1 (int)

    Parameters:
        filename (str): The filename to extract the ID from.
        prefix (str): The prefix to remove (e.g., 'image').
        suffix (str): The suffix to remove (e.g., '.nii').

    Returns:
        int: The numeric ID as an integer.
    """
    numeric_id = filename.replace(prefix, '').replace(suffix, '').lstrip('0')
    return int(numeric_id)  # Convert to integer for comparison


def preprocess_images_to_match_labels(root_dir, output_dir, image_prefix, label_prefix, suffix=".nii"):
    """
    Preprocess all images in train, test, and validation directories to match their labels.

    Parameters:
        root_dir (str): Root directory containing 'train', 'test', and 'validation' directories.
        output_dir (str): Directory to save the adapted images.
        image_prefix (str): Prefix for image filenames (e.g., 'image').
        label_prefix (str): Prefix for label filenames (e.g., 'label').
        suffix (str): Suffix for both filenames (default: '.nii').
    """
    # Splits and subdirectories
    splits = ['train']

    for split in splits:
        image_dir = os.path.join(root_dir, split, 'image')
        label_dir = os.path.join(root_dir, split, 'label')
        output_image_dir = os.path.join(output_dir, split, 'image')
        output_label_dir = os.path.join(output_dir, split, 'label')

        # Ensure output directories exist
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        already_there = os.listdir(f"{output_dir}/{split}/label")
        
        
        # Map labels by numeric ID
        label_files = [f for f in os.listdir(label_dir) if f not in already_there]
        label_map = {
            get_numeric_id(label, label_prefix, suffix): os.path.join(label_dir, label)
            for label in label_files if label.startswith(label_prefix) and label.endswith(suffix)
        }
        

        # Process each image file
        for file_name in os.listdir(image_dir):
            if file_name.startswith(image_prefix) and file_name.endswith(suffix):
                image_id = get_numeric_id(file_name, image_prefix, suffix)
                image_path = os.path.join(image_dir, file_name)
                
                # Match the image to its corresponding label
                if image_id in label_map:
                    label_path = label_map[image_id]
                    output_image_path = os.path.join(output_image_dir, file_name)
                    output_label_path = os.path.join(output_label_dir, os.path.basename(label_path))

                    # Copy label to the output directory without modification
                    if not os.path.exists(output_label_path):
                        label_data, label_affine = load_nifti(label_path)
                        save_nifti(label_data, label_affine, output_label_path)

                    # Adapt image axes to match label
                    adapt_image_axes(image_path, label_path, output_image_path)
                else:
                    print(f"Warning: No matching label found for image {file_name}")


if __name__ == "__main__":
    # Define directories and prefixes
    root_dir = "."  # Replace with your original data path
    output_dir = "swapped_axes"  # Replace with your desired output path
    image_prefix = "image"  # Prefix for image files
    label_prefix = "label"  # Prefix for label files

    # Preprocess the dataset
    preprocess_images_to_match_labels(root_dir, output_dir, image_prefix, label_prefix)