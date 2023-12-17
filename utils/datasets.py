import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """
        A custom dataset class for handling image datasets.

        Attributes:
            images (array): An array of images.
            labels (array): An array of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.

        Methods:
            __len__: Returns the number of items in the dataset.
            __getitem__: Retrieves an image-label pair by index with optional transformations.
        """

    def __init__(self, images, labels, transform=None):

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]
        elif image.shape[-1] == 3:
            image = image.transpose((2, 0, 1))  # Convert from (H,W,C) to (C,H,W)

        image = torch.tensor(image / 255., dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def get_loader(x, y, batch_size, mean, std, flag='Train'):
    """
        Creates and returns a DataLoader for either training or testing.

        Args:
            x (array): Input images.
            y (array): Corresponding labels for the input images.
            batch_size (int): Batch size for the DataLoader.
            mean (float): Mean for normalization.
            std (float): Standard deviation for normalization.
            flag (str): Indicates 'Train' for training data and anything else for testing/validation data.

        Returns:
            DataLoader: A DataLoader object for iterating over the dataset.
        """

    if x.shape[-1] == 3: n = 3
    else: n = 1

    if flag == 'Train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            # transforms.Normalize(mean=[0.568] * n, std=[0.169] * n)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean] * n, std=[std] * n)
        ])

    dataset = CustomDataset(x, y, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(flag == 'Train'))
    return loader


def get_data(data):
    """
       Extracts and returns the training, validation, and testing datasets.

       Args:
           data (dict): A dictionary containing keys 'train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels'.

       Returns:
           tuple: A tuple containing six elements - training images, validation images, test images, training labels, validation labels, and test labels.
       """
    x_train = data['train_images']
    x_val = data['val_images']
    x_test = data['test_images']
    y_train = data['train_labels']
    y_val = data['val_labels']
    y_test = data['test_labels']
    return x_train, x_val, x_test, y_train, y_val, y_test
