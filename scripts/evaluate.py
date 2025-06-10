import torch
from torch.utils.data import DataLoader, Dataset  # Dummy dataset for now
import yaml
import argparse
import os
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.esam2_bls_model import ESAM2_BLS
from src.utils import get_device, set_seed, dice_score  # Add other metrics if needed (e.g., IoU from loss.py)
from src.loss import IoULoss  # For calculating IoU metric


# --- Dummy Dataset (same as in train.py for consistency) ---
class DummyDataset(Dataset):
    def __init__(self, num_samples, img_channels, img_size, num_classes):
        self.num_samples = num_samples
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(self.img_channels, self.img_size, self.img_size)
        mask = (torch.rand(self.num_classes, self.img_size, self.img_size) > 0.7).float()
        return image, mask


def evaluate_model(model, dataloader, device):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    num_samples = 0

    iou_metric_fn = IoULoss()  # IoULoss returns 1 - IoU, so we'll adjust

    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            list_of_output_logits = model(images)
            final_logits = list_of_output_logits[-1]  # Use the final prediction
            final_probs = torch.sigmoid(final_logits)

            # Threshold probabilities to get binary predictions for metrics if needed
            # pred_masks_binary = (final_probs > 0.5).float()

            dice = dice_score(final_probs, masks)  # dice_score expects probabilities

            # Calculate IoU using IoULoss (it calculates 1 - IoU, so subtract from 1)
            # IoULoss expects probabilities
            iou_val = 1.0 - iou_metric_fn(final_probs, masks)

            total_dice += dice.item() * images.size(0)
            total_iou += iou_val.item() * images.size(0)
            num_samples += images.size(0)

            progress_bar.set_postfix(dice=dice.item(), iou=iou_val.item())

    avg_dice = total_dice / num_samples if num_samples > 0 else 0
    avg_iou = total_iou / num_samples if num_samples > 0 else 0

    return avg_dice, avg_iou


def main(config_path, model_weights_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('eval_params', {}).get('seed', 42))  # Use a different seed or same as train
    device = get_device()
    print(f"Using device: {device}")

    # --- Dataloader (Replace with actual test data loading) ---
    print("Setting up dummy test dataloader...")
    test_dataset = DummyDataset(
        num_samples=config.get('data_params', {}).get('num_test_samples', 100),
        img_channels=config['model_params']['img_channels'],
        img_size=config['data_params']['img_size'],
        num_classes=config['model_params']['num_classes']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('eval_params', {}).get('batch_size', config['train_params']['batch_size']),
        shuffle=False,
        num_workers=config.get('eval_params', {}).get('num_workers', config['train_params']['num_workers'])
    )
    print("Test dataloader created.")

    # --- Model ---
    print(f"Initializing ESAM2-BLS model and loading weights from {model_weights_path}...")
    model = ESAM2_BLS(**config['model_params']).to(device)

    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at {model_weights_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print("Model weights loaded.")

    # --- Evaluation ---
    print("Starting evaluation...")
    avg_dice, avg_iou = evaluate_model(model, test_loader, device)

    print("\n--- Evaluation Results ---")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ESAM2-BLS model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration YAML file used for training.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained model weights (.pth file).")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    main(args.config, args.weights)