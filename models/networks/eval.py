import torch
import time
from torchinfo import summary

# Evaluation functions
def pixel_accuracy(pred, target):
    """Compute pixel-wise accuracy."""
    pred = pred.argmax(dim=1)  # Convert logits to class predictions
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return correct / total

def measure_latency(model, input_tensor, device='cuda'):
    """Measure latency of the model."""
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    
    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # Measure over 100 iterations
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 100  # Average per iteration
    return avg_latency * 1000  # Convert to milliseconds

def measure_throughput(model, input_tensor, batch_size=32, device='cuda'):
    """Measure throughput of the model."""
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    
    # Measure throughput
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # Run for 100 batches
            _ = model(input_tensor)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = (batch_size * 100) / total_time  # Samples per second
    return throughput

def measure_flops(model, input_size):
    """Measure FLOPs of the model."""
    return summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )

def evaluate_unet(model, dataloader, device='cuda'):
    """Evaluate U-Net model on validation data and compute metrics."""
    model.eval()
    pixel_acc_total = 0
    num_batches = len(dataloader)
    
    # Iterate over the validation dataset
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute pixel-wise accuracy
            pixel_acc_total += pixel_accuracy(outputs, targets).item()
    
    # Average pixel accuracy
    pixel_acc = pixel_acc_total / num_batches
    return pixel_acc

# Main evaluation script
if __name__ == "__main__":
    import argparse
    from unet import ShortCircuitUNet  # Replace with your model file if different
    from torch.utils.data import DataLoader
    from some_dataset import YourDataset  # Replace with your dataset class

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate U-Net model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the validation dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--image_size', type=int, default=256, help='Image size (assumed square).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run evaluation on (cuda or cpu).')
    args = parser.parse_args()

    # Load model
    model = ShortCircuitUNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    # Load validation dataset
    val_dataset = YourDataset(args.data_path, split='val', transform=None)  # Add necessary transforms
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Input tensor for efficiency metrics
    dummy_input = torch.rand(args.batch_size, 1, args.image_size, args.image_size)

    # Evaluate pixel accuracy
    print("Evaluating pixel-wise accuracy...")
    pixel_accuracy_score = evaluate_unet(model, val_dataloader, device=args.device)
    print(f"Pixel Accuracy: {pixel_accuracy_score:.4f}")

    # Measure latency
    print("Measuring latency...")
    latency = measure_latency(model, dummy_input, device=args.device)
    print(f"Latency: {latency:.2f} ms")

    # Measure throughput
    print("Measuring throughput...")
    throughput = measure_throughput(model, dummy_input, batch_size=args.batch_size, device=args.device)
    print(f"Throughput: {throughput:.2f} samples/sec")

    # Measure FLOPs
    print("Measuring FLOPs...")
    flops = measure_flops(model, (args.batch_size, 1, args.image_size, args.image_size))
    print(f"FLOPs:\n{flops}")
