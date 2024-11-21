import os
import random
from types import SimpleNamespace

import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.utils import save_image


def save_augmented_images(dataset: torch.utils.data.Dataset, name: str = "train", num_images: int = 10) -> None:
    """
    Saves augmented images from a dataset to a specified directory.

    Args:
        dataset (torch.utils.data.Dataset): Dataset containing images and labels.
        name (str): Name prefix for the saved images (default: "train").
        num_images (int): Number of images to save (default: 10).
    """
    save_dir = os.getenv('SAVE_DIR_AUGS')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(num_images):
        scaled_array1, scaled_array2, label = dataset[i]
        save_image(scaled_array1, os.path.join(save_dir, f'augmented_image_{i, name}_1.png'))
        save_image(scaled_array2, os.path.join(save_dir, f'augmented_image_{i, name}_2.png'))


class GaussianNoise:
    """
    Applies Gaussian noise to an image.

    Args:
        noise_variance (Tuple[float, float]): Variance range for the noise (default: (0, 0.1)).
    """

    def __init__(self, noise_variance=(0, 0.1)):
        self.noise_variance = noise_variance

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        # Generate greyscale noise
        noise = torch.randn(img.size(1), img.size(2)) * self.noise_variance[1] + self.noise_variance[0]
        # Add the same noise to each channel
        noisy_img = img + noise.unsqueeze(0).expand_as(img)
        return torch.clamp(noisy_img, 0.0, 1.0)

    def __repr__(self):
        return self.__class__.__name__ + '(noise_variance={0})'.format(self.noise_variance)


class RandomAffineWithProbability:
    """
    Applies random affine transformations with a specified probability.

    Args:
        degrees (float): Range of rotation degrees.
        translate (Optional[Tuple[float, float]]): Range of horizontal and vertical translations.
        scale (Optional[Tuple[float, float]]): Range of scaling factors.
        shear (Optional[Tuple[float, float]]): Range of shearing angles.
        p (float): Probability of applying the transformation (default: 0.5).
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, p=0.5):
        self.affine = transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear, fill=0)
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return self.affine(img)
        return img


class RandomGaussianBlur:
    """
    Applies Gaussian blur to an image with a specified probability.

    Args:
        probability (float): Probability of applying the blur (default: 0.5).
        kernel_size (int): Size of the kernel (default: 5).
        sigma (Tuple[float, float]): Range of standard deviations for the Gaussian kernel.
    """

    def __init__(self, probability=0.5, kernel_size=5, sigma=(0.1, 2.0)):
        self.probability = probability
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < self.probability:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img


class ZScoreNormalization:
    """
    Applies Z-score normalization to an image.
    """

    def __call__(self, img):
        mean = img.mean([1, 2], keepdim=True)
        std = img.std([1, 2], keepdim=True)
        return (img - mean) / std


def get_training_augmentation_scheme(config: SimpleNamespace) -> transforms.Compose:
    """
    Returns the training augmentation scheme based on the specified type.

    Args:
        augmentation_type (str): Type of augmentation ("heavy", "new", "zscore", or default).

    Returns:
        Callable: Composed transformations.
    """
    if config.augment == "heavy":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(config.input_size, config.input_size), scale=(0.8, 1.0),
                                         ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.RandomApply([GaussianNoise(noise_variance=(0, 0.01))], p=0.3),  # Slightly reduce noise variance
            RandomGaussianBlur(probability=0.3, kernel_size=3, sigma=(0.1, 0.8)),  # Slightly reduce blur intensity
            transforms.Normalize(mean=config.norm_mean, std=config.norm_std),
        ])
    elif config.augment == "new":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(config.input_size, config.input_size), scale=(0.8, 1.0),
                                         ratio=(0.9, 1.1)),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.RandomApply([GaussianNoise(noise_variance=(0, 0.05))], p=0.2),  # Slightly reduce noise variance
            RandomGaussianBlur(probability=0.2, kernel_size=3, sigma=(0.9, 1.0)),  # Slightly reduce blur intensity
            transforms.Normalize(mean=config.norm_mean, std=config.norm_std),
        ])
    elif config.augment == "zscore":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(config.input_size, config.input_size), scale=(0.75, 1.0),
                                         ratio=(0.9, 1.1)),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            RandomGaussianBlur(probability=0.3, kernel_size=3, sigma=(0.1, 0.8)),  # Slightly reduce blur intensity
            ZScoreNormalization()
        ])
    elif config.augment == "ae":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            RandomGaussianBlur(probability=0.5, kernel_size=5, sigma=(0.1, 2.0)),
            transforms.Grayscale(num_output_channels=3),  # Ensure image is RGB
            transforms.ToTensor(),
            ZScoreNormalization(),  # Apply Z-score normalization based on each image
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(config.input_size, config.input_size), scale=(0.8, 1.0)),
            # Randomly crop and resize to 224x224
            transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
            transforms.RandomRotation(10),  # Randomly rotate the image
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformations
            RandomGaussianBlur(probability=0.3, kernel_size=5, sigma=(0.1, 1.0)),  # Apply Gaussian Blur
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=config.norm_mean, std=config.norm_std),  # Normalize using ImageNet mean and std
        ])


def get_validation_augmentation_scheme(config: SimpleNamespace) -> transforms.Compose:
    """
    Returns the validation augmentation scheme based on the specified type.

    Args:
        augmentation_type (str): Type of augmentation ("zscore" or default).

    Returns:
        Callable: Composed transformations.
    """
    if config.augment == "zscore":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.input_size),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            ZScoreNormalization()
        ])
    elif config.augment == "ae":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),  # Center crop to ensure consistent size
            transforms.Grayscale(num_output_channels=3),  # Ensure image is RGB
            transforms.ToTensor(),
            ZScoreNormalization(),  # Apply Z-score normalization based on each image
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.input_size),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.norm_mean, std=config.norm_std),
        ])
