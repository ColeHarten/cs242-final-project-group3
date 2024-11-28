import os
import pydicom
import numpy as np
import nibabel as nib  # or use SimpleITK if preferred
from pathlib import Path
import re

def convert_dicom_to_nifti(dicom_dir, output_dir, train_split=62):
    """
    Converts .dcm files to .nii format, splits into train and test sets, 
    and organizes them into separate directories.

    Parameters:
        dicom_dir (str): Root directory containing nested DICOM files.
        output_dir (str): Directory to save train and test .nii files.
        train_split (int): Number of images to use for training.
    """
    # Ensure output directories exist
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Gather all identifiers and sort them
    data_info = []
    for root, _, files in os.walk(dicom_dir):
        dicom_files = [os.path.join(root, f) for f in files if f.endswith('.dcm')]
        if dicom_files:
            # Extract the second outermost directory name (e.g., PANCREAS_0052)
            parts = Path(root).parts
            identifier = parts[-3]  # Assuming "Pancreas-CT/PANCREAS_0052/..."
            numeric_id = re.search(r'\d+', identifier).group().zfill(5)  # Extract numbers and pad to 5 digits
            data_info.append((numeric_id, dicom_files))
    
    # Sort by identifier to ensure reproducibility of train-test split
    data_info.sort(key=lambda x: x[0])

    # Process each group and save to train or test directories
    for i, (numeric_id, dicom_files) in enumerate(data_info):
        print(f"Processing image{numeric_id} with {len(dicom_files)} DICOM files...")
        
        # Read DICOM files and stack them into a 3D volume
        slices = [pydicom.dcmread(f) for f in dicom_files]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Sort by slice location
        volume = np.stack([s.pixel_array for s in slices])

        # Normalize pixel intensities (optional)
        volume = volume.astype(np.int16)

        # Determine output directory (train or test)
        subset_dir = train_dir if i < train_split else test_dir
        nii_file_path = os.path.join(subset_dir, f"image{numeric_id}.nii")

        # Create .nii file and save it
        nii_img = nib.Nifti1Image(volume, np.eye(4))  # Replace np.eye(4) with actual affine matrix if available
        nib.save(nii_img, nii_file_path)
        print(f"Saved {nii_file_path}")

if __name__ == "__main__":
    # Directories
    dicom_dir = "Pancreas-CT"  # Update with your actual DICOM directory path
    output_dir = "output_nii"  # Root directory to save train/test .nii files

    # Convert DICOM to NIfTI and split into train/test
    convert_dicom_to_nifti(dicom_dir, output_dir, train_split=62)
