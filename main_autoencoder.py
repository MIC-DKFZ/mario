import os
import os
import sys
from types import SimpleNamespace
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from DataClass import mAEData
from Model import mAE  # Import the new model
from utils.augmentations import get_training_augmentation_scheme, \
    get_validation_augmentation_scheme
from utils.utils import parse_args_ae, set_environment


def unnormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Reverts normalization applied to an image tensor.

    Args:
        tensor (torch.Tensor): Normalized tensor.
        mean (List[float]): Mean values used for normalization.
        std (List[float]): Standard deviation values used for normalization.

    Returns:
        torch.Tensor: Unnormalized tensor.
    """
    tensor_unnorm = tensor.clone()
    mean = tensor_unnorm.mean([1, 2], keepdim=True)
    std = tensor_unnorm.std([1, 2], keepdim=True)
    for t, m, s in zip(tensor_unnorm, mean, std):
        t.mul_(s).add_(m)
    return tensor_unnorm


def train_mae_model(
        model: mAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        num_epochs: int,
        device: torch.device,
        log_dir: str
) -> None:
    """
    Trains a Masked Autoencoder (mAE) model.

    Args:
        model (mAE): The Masked Autoencoder model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (Adam): Optimizer for training.
        scheduler (ReduceLROnPlateau): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to train on (CPU or GPU).
        log_dir (str): Directory to save logs and model checkpoints.
    """
    train_writer = SummaryWriter(log_dir)
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_model_path = os.path.join(log_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs in tqdm(train_loader):
            imgs = imgs.to(device)

            optimizer.zero_grad()

            with autocast():
                preds, masks = model(imgs)
                loss = model.loss(imgs, preds, masks)  # Call loss with all three arguments

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                preds, masks = model(imgs)
                loss = model.loss(imgs, preds, masks)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        train_writer.add_scalar('Loss/val', val_loss, epoch)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        save_examples(epoch, imgs, log_dir, masks, preds)

    train_writer.close()


def save_examples(epoch: int, imgs: torch.Tensor, log_dir: str, masks: torch.Tensor, preds: torch.Tensor) -> None:
    """
    Saves examples of input, masked, and reconstructed images during training.

    Args:
        epoch (int): Current training epoch.
        imgs (torch.Tensor): Original images.
        log_dir (str): Directory to save images.
        masks (torch.Tensor): Applied masks.
        preds (torch.Tensor): Reconstructed images.
    """
    num_examples = min(len(imgs), 5)  # Save up to 5 pairs
    mean = [49.63750922, 49.63698931, 49.63616761]
    std = [56.09239637, 56.09268942, 56.0924217]
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 5 * num_examples))
    for i in range(num_examples):
        example_input = unnormalize(imgs[i].cpu(), mean, std)
        example_output = unnormalize(preds[i].cpu(), mean, std)
        mask = masks[i].cpu().squeeze().numpy()
        masked_image = example_input.numpy() * (1 - mask)

        axes[i, 0].imshow(example_input[0, :, :], cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask[0, :, :], cmap='gray')
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(masked_image[0, :, :], cmap='gray')
        axes[i, 2].set_title('Masked Image')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(example_output[0, :, :], cmap='gray')
        axes[i, 3].set_title('Reconstruction')
        axes[i, 3].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'epoch_{epoch + 1}.png'))
    plt.close()


def predict(model_path: str, img_path: str, mask_ratios: List[float] = [0.75]) -> None:
    """
    Loads a trained mAE model and performs predictions on a given image.

    Args:
        model_path (str): Path to the trained model file.
        img_path (str): Path to the image file.
        mask_ratios (List[float]): List of mask ratios for testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # Apply transform
    input_image = transform(image).unsqueeze(0).to(device)

    num_of_ratios = len(mask_ratios)
    fig, ax = plt.subplots(num_of_ratios, 4, figsize=(20, 6 * num_of_ratios))

    if num_of_ratios == 1:
        ax = ax.reshape(1, 4)

    for i, mask_ratio in enumerate(mask_ratios):
        model = mAE(in_channels=3, out_channels=3, mask_ratio=mask_ratio, use_mask=(mask_ratio != 0))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = model.to(device)

        with torch.no_grad():
            reconstructed_image, mask = model(input_image)

        mask = mask.squeeze(0).cpu().permute(1, 2, 0).numpy().squeeze()

        input_image_np = input_image.cpu().squeeze(0).cpu()
        reconstructed_image_np = reconstructed_image.cpu().squeeze(0)
        masked_image_np = input_image_np * (1 - mask[:, :, 0])

        ax[i, 0].imshow(input_image_np[0, :, :], cmap='gray')
        ax[i, 0].set_title(f'Original Image: {mask_ratio}')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(mask[:, :, 0], cmap='gray')
        ax[i, 1].set_title(f'Mask: {mask_ratio}')
        ax[i, 1].axis('off')

        ax[i, 2].imshow(masked_image_np[0, :, :], cmap='gray')
        ax[i, 2].set_title('Masked Image')
        ax[i, 2].axis('off')

        ax[i, 3].imshow(reconstructed_image_np[0, :, :], cmap='gray')
        ax[i, 3].set_title('Reconstructed Image with MSE: {:.4f}'.format(
            torch.nn.functional.mse_loss(input_image, reconstructed_image).item()))
        ax[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction.png')
    plt.show()


def get_train_val_splits(data_class: mAEData, val_split_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Splits a dataset into training and validation sets.

    Args:
        data_class (mAEData): Dataset to split.
        val_split_ratio (float): Ratio of validation samples (default: 0.2).

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    val_size = int(val_split_ratio * len(data_class))
    train_size = len(data_class) - val_size
    return random_split(data_class, [train_size, val_size])


if __name__ == "__main__":
    args = parse_args_ae(sys.argv[1:])
    model_name = args.experiment
    data_prefix, results_prefix = set_environment()
    config = SimpleNamespace(**vars(args))
    config.augment = "ae"

    data_path = os.path.join(data_prefix)
    # To generate this file, run utils.find_images.py
    frame = pd.read_csv(os.path.join(results_prefix, 'image_files.csv'))
    cases = np.array(frame['ID'])

    train_transforms = get_training_augmentation_scheme(config)

    val_transforms = get_validation_augmentation_scheme(config)

    data_class = mAEData(cases, frame, data_path, transform=train_transforms)

    train_dataset, val_dataset = get_train_val_splits(data_class)

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = mAE(in_channels=3, out_channels=3, mask_ratio=0.75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    log_dir = os.path.join(results_prefix, model_name, str(args.batch_size))
    train_mae_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=100, device=device,
                    log_dir=log_dir)
    model_path = os.path.join(log_dir, "best_model.pth")
    img_path = "/example/image/path"
    predict(model_path, img_path)
