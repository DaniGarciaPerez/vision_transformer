"""
Author: Dani Garcia

Description: 
  Module for loading datasets (CIFAR100, ImageNet, FashionMnist)
"""

import torch
import torchvision
import torchvision.transforms as transforms


class LoadDataset:
    """
    Class for handling dataset loading and DataLoaders creation.

    attributes:
    ------------
        transform: torchvision.transforms.Compose -> Standardized transformation for dataset items.
        data_path:str -> Path where datasets are stored.
        dataset_name:str -> Name of the dataset to be loaded.
        dataset_classes:dict -> Keys represent the supported datasets and values the associated classes.
    """


    def __init__(self, dataset_name:str="CIFAR100", data_path:str="./data"):
        """
        Initializes the LoadDataset instance.

        params:
        --------
            dataset_name:str -> Name of the dataset to load. Defaults to "CIFAR100".
            data_path:str -> Path for dataset storage. Defaults to "./data".
        """
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_classes = {
                    "CIFAR100": torchvision.datasets.CIFAR100,
                    "ImageNet": torchvision.datasets.ImageNet,
                    "FashionMnist": torchvision.datasets.FashionMNIST
                }
    
    def access_data(self)->tuple:
        """
        Retrieves the specified dataset.

        returns:
        ---------
            (training_set, validation_set):tuple -> Training and validation dataset instances.
        
        raises:
        ---------
            ValueError: If the specified dataset is not supported.
        """

        
        if self.dataset_name not in self.dataset_classes:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        dataset_class = self.dataset_classes[self.dataset_name]

        training_set = dataset_class(
            self.data_path , train=True, transform=self.transform, download=True
        )
        validation_set = dataset_class(
            self.data_path , train=False, transform=self.transform, download=True
        )

        return training_set, validation_set



    def load_data(self, batch_size:int)->tuple:
        """
        Loads the dataset and returns DataLoaders for training and validation sets.

        params:
        ---------
            batch_size:int -> Batch size for both training and validation datasets.

        returns:
        ---------
            (training_set, validation_set):tuple -> Training and validation DataLoaders.
        """
        try:

            training_set, validation_set = self.access_data()

            # Create data loaders for our datasets. Shuffle for training.
            training_loader = torch.utils.data.DataLoader(
                training_set, batch_size=batch_size, shuffle=True
            )
            validation_loader = torch.utils.data.DataLoader(
                validation_set, batch_size=batch_size, shuffle=False
            )

            print(f"Training set has {len(training_set)} instances")
            print(f"Validation set has {len(validation_set)} instances")

            return training_loader, validation_loader

        except Exception as e:
            print(e)
            exit()


