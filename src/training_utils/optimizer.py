"""
Author: Dani Garcia

Description: 
  Module for loading multiple optimizers for PyTorch models.
"""

import torch

class LoadOptimizer:
    """
    A class to load and initialize various PyTorch optimizers.

    attributes:
    -----------
    optimizer:str -> The name of the optimizer to be loaded (default is "ADAM").
    optimizers_list:dict -> A dictionary mapping optimizer names to their optimizer classes.
    learning_rate:float -> The learning rate for the optimizer (default is 0.003).
    model_parameters:dict -> The parameters of the PyTorch model for which the optimizer is being loaded.

    """

    def __init__(self, model_parameters: dict, optimizer: str = "ADAM", learning_rate: float = 0.003):
        """
        Initializes the LoadOptimizer instance.

        raises:
        ------
            KeyError : If the specified optimizer is not supported.
        """
        self.optimizer = optimizer.upper()
        
        self.optimizers_list = {
            "ADAM": torch.optim.Adam
        }
        self.learning_rate = learning_rate
        self.model_parameters = model_parameters
        
        # Check if the specified optimizer is supported
        if self.optimizer not in self.optimizers_list:
            raise KeyError(f"Unsupported optimizer: {optimizer}. Supported optimizers: {list(self.optimizers_list.keys())}")

    def get_optimizer(self):
        """
        Returns an instance of the specified optimizer, initialized with the model parameters and learning rate.

        returns:
        --------
        torch.optim.Optimizer : The initialized optimizer instance.
        """

        return self.optimizers_list[self.optimizer](self.model_parameters, lr=self.learning_rate)