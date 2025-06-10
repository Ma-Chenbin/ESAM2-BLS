import torch
import random
import numpy as np
import os


def get_device():
    """Returns the appropriate device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dir_if_not_exists(path):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")


# Example of a simple metric function (Dice Score)
def dice_score(pred_probs, target_mask, epsilon=1e-6):
    """
    Computes the Dice score.
    pred_probs: probabilities (after sigmoid), shape (B, C, H, W)
    target_mask: ground truth mask, shape (B, C, H, W)
    """
    assert pred_probs.shape == target_mask.shape
    dims = tuple(range(2, pred_probs.dim()))  # Sum over H, W (and C if not binary)

    intersection = torch.sum(pred_probs * target_mask, dim=dims)
    sum_pred = torch.sum(pred_probs, dim=dims)
    sum_target = torch.sum(target_mask, dim=dims)

    dice = (2. * intersection + epsilon) / (sum_pred + sum_target + epsilon)
    return dice.mean()  # Average Dice over batch and channels


if __name__ == '__main__':
    print(f"Using device: {get_device()}")
    set_seed(123)
    print("Seed set to 123.")

    # Test dice score
    B, C, H, W = 2, 1, 64, 64
    dummy_pred_probs = torch.rand(B, C, H, W)
    dummy_target_mask = (torch.rand(B, C, H, W) > 0.5).float()
    dice = dice_score(dummy_pred_probs, dummy_target_mask)
    print(f"Dummy Dice Score: {dice.item()}")