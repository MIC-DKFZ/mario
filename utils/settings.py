from types import SimpleNamespace
from typing import Tuple, Optional

import torch
from loss import get_loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from Model import get_model
from utils.utils import calculate_balanced_class_weights


def init_settings(
        config: SimpleNamespace,
        train_loader: torch.utils.data.DataLoader,
) -> Tuple[torch.nn.Module, Optimizer, _LRScheduler, Optional[torch.Tensor], torch.nn.Module]:
    """
    Initializes the model, optimizer, scheduler, and loss function for training.

    Args:
        config (SimpleNamespace): Configuration object containing model and training parameters.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.

    Returns:
        Tuple[torch.nn.Module, Optimizer, _LRScheduler, Optional[torch.Tensor], torch.nn.Module]:
            - model: Initialized and moved to CUDA.
            - optimizer: Optimizer for training.
            - scheduler: Learning rate scheduler.
            - class_weights: Class weights for the loss function (or None).
            - criterion: Loss function.
    """
    model = get_model(config)
    print(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay_classifier)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=config.num_epochs)
    class_weights = calculate_balanced_class_weights(dataloader=train_loader,
                                                     num_classes=4) if config.class_weights else None
    if class_weights != None: class_weights = class_weights.cuda()
    criterion = get_loss(config, class_weights=class_weights)
    return model, optimizer, scheduler, class_weights, criterion


def freeze_encoder(model: torch.nn.Module) -> None:
    """
    Freezes the encoder layers of the model to prevent updates during backpropagation.

    Args:
        model (torch.nn.Module): The model whose encoder layers need to be frozen.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False


def init_optimizer(
        config: SimpleNamespace,
) -> Tuple[Optimizer, _LRScheduler]:
    """
    Initializes the optimizer and learning rate scheduler based on the configuration.

    Args:
        config (SimpleNamespace): Configuration object containing optimizer and scheduler parameters.

    Returns:
        Tuple[Optimizer, _LRScheduler]:
            - optimizer: Optimizer for training.
            - scheduler: Learning rate scheduler.
    """
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': config.initial_encoder_lr},
        {'params': model.classifier.parameters(), 'lr': config.initial_classifier_lr,
         'weight_decay': config.weight_decay_classifier}])
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5,
                                                               min_lr=1e-6)
    elif config.scheduler == "factor":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.9, total_iters=config.num_epochs)
    return optimizer, scheduler
