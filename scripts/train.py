import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset  # Dummy dataset for now
import yaml
import argparse
import os
import time
from tqdm import tqdm

# Assuming model.py and loss.py are in src directory, and train.py is in scripts/
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add project root to Python path

from src.esam2_bls_model import ESAM2_BLS  # Make sure to have model.py in src
from src.loss import JointOptimizationLoss
from src.utils import get_device, set_seed, create_dir_if_not_exists, dice_score


# --- Dummy Dataset and DataLoader ---
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
        mask = (torch.rand(self.num_classes, self.img_size, self.img_size) > 0.7).float()  # Sparse masks
        return image, mask


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num, num_epochs):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num + 1}/{num_epochs}", unit="batch")

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        list_of_output_logits = model(images)  # Model returns a list of logits
        loss = criterion(list_of_output_logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate Dice for the final prediction
        final_probs = torch.sigmoid(list_of_output_logits[-1])
        dice = dice_score(final_probs, masks)
        running_dice += dice.item() * images.size(0)

        progress_bar.set_postfix(loss=loss.item(), dice=dice.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    return epoch_loss, epoch_dice


def validate_one_epoch(model, dataloader, criterion, device, epoch_num, num_epochs):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num + 1}/{num_epochs} [Validation]", unit="batch")

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            list_of_output_logits = model(images)
            loss = criterion(list_of_output_logits, masks)

            running_loss += loss.item() * images.size(0)

            final_probs = torch.sigmoid(list_of_output_logits[-1])
            dice = dice_score(final_probs, masks)
            running_dice += dice.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item(), dice=dice.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    return epoch_loss, epoch_dice


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['train_params']['seed'])
    device = get_device()
    print(f"Using device: {device}")

    # --- Dataloaders (Replace with actual data loading) ---
    print("Setting up dummy dataloaders...")
    train_dataset = DummyDataset(
        num_samples=config['data_params']['num_train_samples'],
        img_channels=config['model_params']['img_channels'],
        img_size=config['data_params']['img_size'],
        num_classes=config['model_params']['num_classes']
    )
    val_dataset = DummyDataset(
        num_samples=config['data_params']['num_val_samples'],
        img_channels=config['model_params']['img_channels'],
        img_size=config['data_params']['img_size'],
        num_classes=config['model_params']['num_classes']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=True,
                              num_workers=config['train_params']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch_size'], shuffle=False,
                            num_workers=config['train_params']['num_workers'])
    print("Dataloaders created.")

    # --- Model ---
    print("Initializing ESAM2-BLS model...")
    model = ESAM2_BLS(**config['model_params']).to(device)
    print("Model initialized.")

    # --- Loss Function ---
    # L=3 for deep supervision and specific alpha weights.
    # Lambda weights for the joint loss are hyperparameters.
    print("Initializing loss function...")
    criterion = JointOptimizationLoss(
        lambda_iou=config['loss_params']['lambda_iou'],
        lambda_bce=config['loss_params']['lambda_bce'],
        lambda_ds=config['loss_params']['lambda_ds'],
        ds_num_layers=config['model_params'].get('deep_supervision_levels', 3),  # Ensure this matches model
        ds_weights=config['loss_params'].get('ds_weights', [0.2, 0.3, 0.5])
    ).to(device)
    print("Loss function initialized.")

    # --- Optimizer and Scheduler ---
    # AdamW optimizer and cosine annealing learning rate.
    print("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer_params']['learning_rate'],
        weight_decay=config['optimizer_params']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['scheduler_params'],
        eta_min=config['scheduler_params']['eta_min']
    )
    print("Optimizer and scheduler set up.")

    # --- Training Loop ---
    num_epochs = config['train_params']['num_epochs']
    output_dir = config['train_params']['output_dir']
    create_dir_if_not_exists(output_dir)

    best_val_dice = 0.0
    start_time = time.time()
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, epoch, num_epochs)

        scheduler.step()  # Step the scheduler

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_duration:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
              f"LR: {optimizer.param_groups['lr']:.6f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            model_save_path = os.path.join(output_dir,
                                           f"esam2_bls_best_model_epoch_{epoch + 1}_dice_{val_dice:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path} with Val Dice: {best_val_dice:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % config['train_params'].get('save_checkpoint_freq', 5) == 0:
            checkpoint_path = os.path.join(output_dir, f"esam2_bls_checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'dice': val_dice
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    total_training_time = time.time() - start_time
    print(
        f"Training finished. Total time: {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Best validation Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESAM2-BLS model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    main(args.config)