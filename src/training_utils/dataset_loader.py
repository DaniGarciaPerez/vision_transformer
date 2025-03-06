"""
Author: Dani Garcia

Description: TODO: Insert

"""

import torch
import torchvision
import torchvision.transforms as transforms


class LoadDataset:
    """
    TODO: Insert docstring
    """

    def __init__(self):
        """
        TODO: Insert docstring
        """
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def load_cifar100(self):
        """
        TODO: Insert docstring
        """

        # Create datasets for training & validation, download if necessary
        training_set = torchvision.datasets.CIFAR100(
            "./data", train=True, transform=self.transform, download=True
        )
        validation_set = torchvision.datasets.CIFAR100(
            "./data", train=False, transform=self.transform, download=True
        )

        return training_set, validation_set

    def load_data(self, batch_size, dataset_name="CIFAR100"):
        """
        TODO: Insert docstring
        """

        try:
            if dataset_name == "CIFAR100":
                training_set, validation_set = self.load_cifar100()

            if dataset_name == "ImageNet":
                training_set, validation_set = self.load_cifar100()

        except Exception as e:
            print(e)
            print(
                "Please, select a dataset from the following list: CIFAR100, InageNet"
            )

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(
            training_set, batch_size=batch_size, shuffle=True
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=batch_size, shuffle=False
        )

        # Report split sizes
        print("Training set has {} instances".format(len(training_set)))
        print("Validation set has {} instances".format(len(validation_set)))

        return training_loader, validation_loader
