"""
Author: Dani Garcia

Description: This script implements the image split and first
linear layer to get crops for the image to process.

"""

import cv2 as cv
import torch
from torch import nn


class LinearPatchProjection(nn.Module):

    def __init__(self):
        """
        Initializes the LinearPatchProjection module.

        params:
        """
        # Instantiate the base class
        super(LinearPatchProjection, self).__init__()

    def split_image_patches(self, image: str):
        """TODO: Add docstring"""

        # Read the image and trasnform to a torch tensor
        image_matrix = torch.from_numpy(cv.imread(image))

        torch.tensor_split(image_matrix, (4, 4))

        print(image_matrix.shape)
        print(type(image_matrix))
