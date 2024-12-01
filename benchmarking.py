import numpy, sys
from datetime import datetime as t
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model

# Turn off the deprecations warnings for now
import warnings
warnings.filterwarnings("ignore")

def test_baseline_model(pretrained_model_path):
    json_filename = "configs/config_unet_ct_multi_att_dsv.json"
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model, verbose=False)

    # Load the pretrained model
    model.load_network_from_path(model.net, pretrained_model_path, strict=True)
        
    # Setup Data Loader
    test_dataset = ds_class(
        ds_path, split='test', transform=ds_transform['valid'], preload_data=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False
    )

    # Set the model to evaluation mode
    model.net.eval()

    # Initialize the error logger
    error_logger = ErrorLogger()

    # Testing Iterations
    print("Starting testing...")
    
    start = t.now()
    
    total_flops = 0
    
    niters = 100
    
    for _ in range(niters):
        for images, labels in tqdm(test_loader, total=len(test_loader), file=sys.stdout):
            # Move data to the appropriate device
            if model.use_cuda and model.gpu_ids:
                images, labels = images.cuda(model.gpu_ids[0]), labels.cuda(model.gpu_ids[0])

            # Perform forward pass
            model.set_input(images, labels)
            model.validate()


            # Log errors and stats
            errors = model.get_current_errors()
            stats = model.get_segmentation_stats()
            error_logger.update({**errors, **stats}, split='test')
        
    end = t.now()
    
    print(f"Inference took an average of {(end-start)/(len(test_loader)*niters)} seconds per image.")
    
    # Summarize results
    test_errors = error_logger.get_errors('test')
    print("--------------------------------------")
    print(f"Test Results: {test_errors}")
    print("--------------------------------------")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmarker')
    parser.add_argument('-f', '--file', help='Path to saved model', required=True)

    args = parser.parse_args()
    file = args.file

    test_baseline_model(file)

