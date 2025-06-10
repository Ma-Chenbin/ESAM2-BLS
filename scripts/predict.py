import torch
import yaml
import argparse
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.esam2_bls_model import ESAM2_BLS
from src.utils import get_device, create_dir_if_not_exists


def load_image(image_path, img_size):
    """Loads and preprocesses an image."""
    try:
        img = Image.open(image_path).convert("RGB")  # Assuming 3-channel input
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

    preprocess = T.Compose(std = [0.229, 0.224, 0.225])  # Example normalization
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def save_mask(mask_tensor, output_path, original_size=None):
    """Saves the predicted mask as an image."""
    mask_np = mask_tensor.squeeze().cpu().numpy()  # Remove batch and channel (if 1)

    # Assuming binary mask after thresholding
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')  # Grayscale

    if original_size:
        mask_img = mask_img.resize(original_size, Image.NEAREST)

    try:
        mask_img.save(output_path)
        print(f"Predicted mask saved to {output_path}")
    except Exception as e:
        print(f"Error saving mask to {output_path}: {e}")


def predict_single_image(model, image_tensor, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        list_of_output_logits = model(image_tensor)
        final_logits = list_of_output_logits[-1]  # Use the final prediction
        final_probs = torch.sigmoid(final_logits)

        # Apply threshold to get binary mask
        pred_mask_binary = (final_probs > threshold).float()

    return pred_mask_binary


def main(config_path, model_weights_path, input_path, output_dir, threshold):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    img_size = config['data_params']['img_size']

    # --- Model ---
    print(f"Initializing ESAM2-BLS model and loading weights from {model_weights_path}...")
    model = ESAM2_BLS(**config['model_params']).to(device)

    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at {model_weights_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print("Model weights loaded.")

    create_dir_if_not_exists(output_dir)

    if os.path.isfile(input_path):
        image_paths = [input_path]
    elif os.path.isdir(input_path):
        image_paths = [os.path.join(input_path, fname) for fname in os.listdir(input_path)
                       if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    else:
        print(f"Error: Input path {input_path} is not a valid file or directory.")
        sys.exit(1)

    if not image_paths:
        print(f"No images found in {input_path}.")
        return

    print(f"Found {len(image_paths)} images for prediction.")
    for image_path in image_paths:
        print(f"\nProcessing {image_path}...")

        # Get original image size for resizing mask later
        try:
            with Image.open(image_path) as img_orig:
                original_w, original_h = img_orig.size
        except Exception as e:
            print(f"Could not read original image size for {image_path}: {e}. Mask will not be resized to original.")
            original_w, original_h = None, None

        image_tensor = load_image(image_path, img_size)
        if image_tensor is None:
            continue

        predicted_mask = predict_single_image(model, image_tensor, device, threshold)

        base_filename = os.path.splitext(os.path.basename(image_path))
        output_mask_path = os.path.join(output_dir, f"{base_filename}_mask.png")

        save_mask(predicted_mask, output_mask_path, original_size=(original_w, original_h) if original_w else None)

    print("\nPrediction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict segmentation masks using a trained ESAM2-BLS model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration YAML file used for training.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained model weights (.pth file).")
    parser.add_argument("--input", type=str, required=True, help="Path to an input image or a directory of images.")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save predicted masks.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    main(args.config, args.weights, args.input, args.output_dir, args.threshold)