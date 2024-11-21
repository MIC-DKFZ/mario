import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across libraries and frameworks.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def str2bool(value: str) -> bool:
    """
    Converts a string to a boolean value.

    Args:
        value (str): Input string representing a boolean.

    Returns:
        bool: Converted boolean value.

    Raises:
        ValueError: If the input value cannot be interpreted as a boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes', 'y'}:
        return True
    elif value.lower() in {'false', '0', 'no', 'n'}:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    """
    Parses command-line arguments.

    Args:
        argv (List[str]): List of arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default='Local')
    parser.add_argument('--experiment', type=str, default='exp303')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    parser.add_argument('--scheduler', type=str, default="plateau", help='plateau, cosine, None')
    parser.add_argument('--loss', type=str, default="cross-entropy", help='focal')
    parser.add_argument('--class_weights', type=bool, default=False, help='Manual class weights for the loss function')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='0; Label smoothing factor')

    parser.add_argument('--augment', type=str, default='new', help='light, heavy')
    parser.add_argument('--input_size', type=int, default=224, help='Input size of the images')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406],
                        help='[0.485, 0.456, 0.406]449 Normalization mean for the dataset')
    parser.add_argument('--norm_std', type=float, nargs=3, default=[0.229, 0.224, 0.225],
                        help='[0.229, 0.224, 0.225]226 Normalization standard deviation for the dataset')

    parser.add_argument('--freeze_epoch', type=float, default=10, help='Initial learning rate')
    parser.add_argument('--initial_lr', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--initial_encoder_lr', type=float, default=1e-4, help='Initial learning rate for the encoder')
    parser.add_argument('--initial_classifier_lr', type=float, default=1e-3,
                        help='Initial learning rate for the classifier')
    parser.add_argument('--weight_decay_classifier', type=float, default=0, help='Weight decay for the classifier')

    parser.add_argument('--encoder_weights_path', type=str, default="",
                        help="/home/a332l/E132-Projekte/Projects/2024_MICCAI_Mario_Challenge/data/Runs_Max/ContrastivePretraining/resnet50/checkpoints/model.pth")
    parser.add_argument('--res50', type=str2bool, default="True", help='True uses resnet50, else resnet18')
    parser.add_argument('--sub', type=str2bool, default="True",
                        help='True uses feature vector subtraction instead of concatenation.')
    parser.add_argument('--biomed', type=str2bool, default="False",
                        help='True uses feature vector subtraction instead of concatenation.')
    parser.add_argument('--RETFound', type=str2bool, default="False",
                        help='True uses feature vector subtraction instead of concatenation.')
    parser.add_argument('--mAE_pretrained', type=str2bool, default="False", help='True uses mAE as a pretrained model.')
    parser.add_argument('--contrastive_pretrained', type=str2bool, default="False",
                        help='True uses mAE as a pretrained model.')

    return parser.parse_args(argv)


def parse_args_ae(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='ae')
    parser.add_argument('--environment', type=str, default='Local')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=16)
    return parser.parse_args(argv)


def set_environment() -> Tuple[str, str]:
    """
    Retrieves environment paths from environment variables.

    Returns:
        Tuple[str, str]: Paths for data and results directories.
    """
    data_prefix = os.getenv('DATA_DIR')
    results_prefix = os.getenv('SAVE_DIR_RESULTS')
    return data_prefix, results_prefix


def calculate_balanced_class_weights(
        dataloader: torch.utils.data.DataLoader,
        num_classes: int,
        alpha: float = 0.4
) -> torch.Tensor:
    """
        Calculates class weights for balancing imbalanced datasets.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
            num_classes (int): Number of classes in the dataset.
            alpha (float): Balancing factor to adjust weight uniformity.

        Returns:
            torch.Tensor: Calculated class weights.
        """
    # higher alpha leads to more equal weights independent of class distribution, lower allows for greater difference in the weights
    class_counts = torch.zeros(num_classes)
    for _, _, labels in dataloader:
        labels = np.argmax(labels, axis=1)
        for label in labels:
            class_counts[label] += 1

    total_samples = class_counts.sum()
    class_weights = (1.0 / class_counts) * (total_samples / num_classes)
    class_weights = alpha + (1 - alpha) * class_weights
    return class_weights / class_weights.sum() * num_classes
