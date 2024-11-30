import numpy
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm


from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils import error_logger
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model

from collections import defaultdict

def train(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model)
    try:
        model.set_thresholds('thresholds.pt')
    except FileNotFoundError:
        print("Thresholds not found. Training will proceed without thresholds.")

    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData)
    valid_dataset = ds_class(ds_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    test_dataset  = ds_class(ds_path, split='test',       transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    train_loader = DataLoader(dataset=train_dataset, num_workers=1, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=1, batch_size=train_opts.batchSize, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=1, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    # visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()


    # Initialize storage for class probabilities at each layer
    layer_probs = defaultdict(lambda: defaultdict(list))

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations (Lei Temporarily Modified)
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Collect probabilities for each layer
            with torch.no_grad():
                for layer_name, feature_map in model.layer_outputs.items():
                    probs = F.softmax(feature_map, dim=1)  # Convert logits to probabilities
                    for cls in range(probs.shape[1]):  # Iterate over classes
                        class_mask = (labels == cls)
                        if class_mask.any():
                            layer_probs[layer_name][cls].append(probs[:, cls][class_mask].mean().item())

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                # Visualise predictions
                # visuals = model.get_current_visuals()
                # visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        print("--------------------------------------")
        for split in ['train', 'validation', 'test']:
            epoch_errors = error_logger.get_errors(split)
            print(f"Epoch {epoch} {split} errors: {epoch_errors}")
        print("--------------------------------------")
        
        # Update the plots
        # for split in ['train', 'validation', 'test']:
            # visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            # visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()

    # If thresholds haven't been calculated yet
    if model.thresholds is None:
        # Average probabilities across all layers
        class_mean_probs = defaultdict(list)
        for cls in range(model.n_classes):
            layer_means = []
            for layer_name in layer_probs:
                if len(layer_probs[layer_name][cls]) > 0:
                    layer_means.append(sum(layer_probs[layer_name][cls]) / len(layer_probs[layer_name][cls]))
            if layer_means:
                class_mean_probs[cls] = sum(layer_means) / len(layer_means)

        # Compute and scale thresholds
        thresholds = []
        alpha, beta = 0.95, 0.998 # hyperparameters for threshold scaling
        for cls in range(model.n_classes):
            if cls not in class_mean_probs or not class_mean_probs[cls]:
                thresholds.append(0)
            else:
                sorted_probs = sorted(class_mean_probs[cls], reverse=True)
                T_k = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
                thresholds.append(T_k)

        min_T, max_T = min(thresholds), max(thresholds)
        scaled_thresholds = [
            (1 - (T_k - min_T) / (max_T - min_T)) * (beta - alpha) + alpha if max_T > min_T else alpha
            for T_k in thresholds
        ]

        try:
            torch.save(scaled_thresholds, 'thresholds.pt')
        except FileNotFoundError:
            print("Thresholds could not be saved.")
    
        print(f"Scaled thresholds saved: {scaled_thresholds}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
