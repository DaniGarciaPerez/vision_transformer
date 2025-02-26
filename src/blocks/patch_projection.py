"""
Author: Dani Garcia

Description: This script implements the image split and first
linear layer to get crops for the image to process.

"""

import torch
import cv2
from torch import nn


class PatchLinearProjection(nn.Module):
    """TODO: Add docstring"""

    def __init__(self, num_patches, d_model):
        """
        Initializes the LinearPatchProjection module.

        params:
        """
        # Instantiate the base class
        super(PatchLinearProjection, self).__init__()
        self.num_patches = num_patches
        self.d_model = d_model
        self.linear_weights = nn.Linear(num_patches, d_model)

    def split_image_patches(self, image):
        """TODO: Add docstring"""

        # Read the image and trasnform to a torch tensor with shape CxHxW
        image_matrix = (
            torch.from_numpy(cv2.imread(image)).to(torch.float32).movedim(2, 0)
        )

        kernel_size = int(
            torch.sqrt(
                torch.tensor(
                    image_matrix.shape[2] * image_matrix.shape[1] // self.num_patches
                )
            )
        )
        print(kernel_size)

        unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=kernel_size)
        unfolded_image = torch.transpose(unfold(image_matrix), 0, 1)

        return torch.nn.functional.normalize(unfolded_image)

    def forward(self, image_to_transform) -> torch.tensor:
        """TODO: Add docstring"""

        image_patches = self.split_image_patches(image_to_transform)

        return self.linear_weights(image_patches)
