from types import SimpleNamespace

import torch


def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1,
        gamma: float = 2,
        reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes the focal loss for a classification task.

    Args:
        inputs (torch.Tensor): Model predictions (logits).
        targets (torch.Tensor): Ground truth labels.
        alpha (float): Weighting factor for the positive class (default=1).
        gamma (float): Focusing parameter to reduce the impact of well-classified examples (default=2).
        reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: Computed focal loss.
    """
    BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduction == 'mean':
        return F_loss.mean()
    elif reduction == 'sum':
        return F_loss.sum()
    else:
        return F_loss


def get_loss(config: SimpleNamespace, class_weights: torch.Tensor):
    """
    Returns the appropriate loss function based on the configuration.

    Args:
        config (object): Configuration object containing loss parameters.

    Returns:
        Loss function to be used during training.
    """
    if config.loss == "focal":
        return lambda inputs, targets: focal_loss(inputs, targets, alpha=1, gamma=2, reduction='mean')
    else:
        return torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.label_smoothing,
                                         weight=class_weights)
