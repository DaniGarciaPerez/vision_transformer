"""
Author: Dani Garcia

Description: This script implements the image split and first
linear layer to get crops for the image to process.

"""

import cv2 as cv2
import torch
from torch import nn
from torchvision import transforms
import numpy as np


class PatchLinearProjection(nn.Module):

    def __init__(self):
        """
        Initializes the LinearPatchProjection module.

        params:
        """
        # Instantiate the base class
        super(PatchLinearProjection, self).__init__()

    def split_image_patches(self, image: str):
        """TODO: Add docstring"""

        # Read the image and trasnform to a torch tensor with shape CxHxW
        image_matrix = (
            torch.from_numpy(cv2.imread(image)).to(torch.float32).movedim(2, 0)
        )

        size = 18
        unfold = nn.Unfold(kernel_size=(size, size), stride=size)
        unfolded_image = torch.transpose(unfold(image_matrix), 0, 1)

        return torch.nn.functional.normalize(unfolded_image)
