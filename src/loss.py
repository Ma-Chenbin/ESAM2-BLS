import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss.
    As defined in Equation 12 of the ESAM2-BLS.
    $L_{IoU} = 1 - \frac{\sum_{i=1}^{N} p_i g_i}{\sum_{i=1}^{N} (p_i + g_i - p_i g_i)}$
    """

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_probs, target_mask):
        # pred_probs should be after sigmoid
        assert pred_probs.shape == target_mask.shape, \
            f"Input and target shapes must match. Got {pred_probs.shape} and {target_mask.shape}"

        dims = tuple(range(1, pred_probs.dim()))  # Sum over all dims except batch

        intersection = torch.sum(pred_probs * target_mask, dim=dims)
        union = torch.sum(pred_probs, dim=dims) + torch.sum(target_mask, dim=dims) - intersection

        iou = (intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - iou.mean()


class BCELossWrapper(nn.Module):
    """
    Binary Cross-Entropy (BCE) Loss.
    As defined in Equation 13 of the ESAM2-BLS.
    $L_{BCE} = - \frac{1}{N} \sum_{i=1}^{N} [g_i \log(p_i) + (1 - g_i) \log(1 - p_i)]$
    This wrapper uses BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_mask):
        assert pred_logits.shape == target_mask.shape, \
            f"Input and target shapes must match. Got {pred_logits.shape} and {target_mask.shape}"
        return self.bce_with_logits(pred_logits, target_mask)


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss.
    As defined in Equation 14 of the ESAM2-BLS.
    $L_{DS} = \sum_{l=1}^{L} \alpha_l (L_{IoU}(\hat{y}_l, y) + L_{BCE}(\hat{y}_l, y))$
    Weights (alpha_l) for L=3 layers are: alpha_1=0.2, alpha_2=0.3, alpha_3=0.5.
    """

    def __init__(self, num_supervised_layers=3, weights=None):
        super().__init__()

        if weights is None:
            # Default L=3
            if num_supervised_layers == 3:
                self.weights = [0.2, 0.3, 0.5]
            else:
                # Fallback to equal weighting if num_supervised_layers is different
                # and no specific weights are provided.
                print(
                    f"Warning: Using equal weights for DeepSupervisionLoss as num_supervised_layers is {num_supervised_layers} and no specific weights were given.")
                self.weights = [1.0 / num_supervised_layers] * num_supervised_layers
        else:
            if len(weights) != num_supervised_layers:
                raise ValueError(
                    f"Length of weights ({len(weights)}) must match num_supervised_layers ({num_supervised_layers})")
            self.weights = weights

        self.num_supervised_layers = num_supervised_layers
        self.iou_loss = IoULoss()
        self.bce_loss = BCELossWrapper()

    def forward(self, pred_logits_list, target_mask):
        total_ds_loss = 0.0

        num_predictions = len(pred_logits_list)
        if num_predictions != self.num_supervised_layers:
            print(f"Warning: Number of predicted masks ({num_predictions}) "
                  f"does not match configured number of supervised layers ({self.num_supervised_layers}). "
                  f"Will use {min(num_predictions, self.num_supervised_layers)} layers for DS loss.")

        num_to_supervise = min(num_predictions, self.num_supervised_layers)

        for i in range(num_to_supervise):
            # pred_logits_list is ordered from shallowest supervised to deepest.
            # self.weights are also ordered shallowest to deepest.
            logits = pred_logits_list[i]
            probs = torch.sigmoid(logits)  # IoULoss expects probabilities

            loss_iou_i = self.iou_loss(probs, target_mask)
            loss_bce_i = self.bce_loss(logits, target_mask)  # BCELossWrapper takes logits

            layer_loss = self.weights[i] * (loss_iou_i + loss_bce_i)
            total_ds_loss += layer_loss

        return total_ds_loss / num_to_supervise if num_to_supervise > 0 else torch.tensor(0.0,
                                                                                          device=target_mask.device)


class JointOptimizationLoss(nn.Module):
    """
    Total Loss Function for ESAM2-BLS.
    As defined in Equation 15 of the ESAM2-BLS.
    $L_{total} = \lambda_1 L_{IoU} + \lambda_2 L_{BCE} + \lambda_3 L_{DS}$
    Lambda coefficients are hyperparameters.
    """

    def __init__(self, lambda_iou=1.0, lambda_bce=1.0, lambda_ds=1.0,
                 ds_num_layers=3, ds_weights=None):
        super().__init__()
        self.lambda_iou = lambda_iou
        self.lambda_bce = lambda_bce
        self.lambda_ds = lambda_ds

        self.main_iou_loss = IoULoss()
        self.main_bce_loss = BCELossWrapper()
        self.ds_loss_module = DeepSupervisionLoss(num_supervised_layers=ds_num_layers, weights=ds_weights)

    def forward(self, pred_logits_list, target_mask):
        # pred_logits_list is ordered shallowest to deepest supervised.
        # The last one is the primary prediction.
        if not pred_logits_list:
            raise ValueError("pred_logits_list cannot be empty.")

        final_logits = pred_logits_list[-1]
        final_probs = torch.sigmoid(final_logits)

        loss_main_iou = self.main_iou_loss(final_probs, target_mask)
        loss_main_bce = self.main_bce_loss(final_logits, target_mask)

        loss_ds = self.ds_loss_module(pred_logits_list, target_mask)

        total_loss = (self.lambda_iou * loss_main_iou +
                      self.lambda_bce * loss_main_bce +
                      self.lambda_ds * loss_ds)
        return total_loss


if __name__ == '__main__':
    # Example Usage
    num_classes = 1
    img_size = 64  # Small for testing
    batch_size = 2
    deep_supervision_levels = 3

    # Dummy model outputs (list of logits for deep supervision)
    dummy_outputs = 2
    dummy_target = (torch.rand(batch_size, num_classes, img_size, img_size) > 0.5).float()

    # Test IoULoss
    iou_loss_fn = IoULoss()
    loss_iou_val = iou_loss_fn(torch.sigmoid(dummy_outputs[-1]), dummy_target)
    print(f"IoU Loss: {loss_iou_val.item()}")

    # Test BCELossWrapper
    bce_loss_fn = BCELossWrapper()
    loss_bce_val = bce_loss_fn(dummy_outputs[-1], dummy_target)
    print(f"BCE Loss: {loss_bce_val.item()}")

    # Test DeepSupervisionLoss
    ds_weights = [0.2, 0.3, 0.5]
    ds_loss_fn = DeepSupervisionLoss(num_supervised_layers=deep_supervision_levels, weights=ds_weights)
    loss_ds_val = ds_loss_fn(dummy_outputs, dummy_target)
    print(f"Deep Supervision Loss: {loss_ds_val.item()}")

    # Test JointOptimizationLoss
    # Lambda weights are hyperparameters
    joint_loss_fn = JointOptimizationLoss(
        lambda_iou=1.0, lambda_bce=1.0, lambda_ds=1.0,
        ds_num_layers=deep_supervision_levels,
        ds_weights=ds_weights
    )
    total_loss_val = joint_loss_fn(dummy_outputs, dummy_target)
    print(f"Joint Optimization Loss: {total_loss_val.item()}")