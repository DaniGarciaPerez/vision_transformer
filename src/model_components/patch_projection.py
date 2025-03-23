"""
Author: Dani Garcia

Description: This script implements the image split and first
linear layer to get crops for the image to process.

"""

import torch
from torch import nn


class PatchLinearProjection(nn.Module):
    """
    A module that applies a linear projection to image patches.

    This module splits an input image into patches, applies a linear transformation
    to each patch, and returns the projected patches.

    attributes:
    ------------
        patch_size:int -> The size of each patch.
        d_model:int -> The number of output features for each patch.
        linear_weights:nn.Conv2d -> The linear weights for the projection.
    """

    def __init__(self, patch_size: int, d_model: int, input_channels: int = 3):
        """
        Initializes the PatchLinearProjection module.

        params:
        -------
            patch_size:int -> The size of each patch.
            d_model:int -> The number of output features for each patch.
        """
        # Instantiate the base class
        super(PatchLinearProjection, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        # Initialize the linear weights as a convolutional layer
        # with a kernel size equal to the patch size
        self.linear_weights = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear projection to the input image.

        params:
        --------
            image_tensor:torch.Tensor -> Image tensor

        returns:
        ---------
            torch.Tensor -> The projected patches.
        """

        return self.linear_weights(image_tensor).flatten(2, 3).movedim(1, -1)
