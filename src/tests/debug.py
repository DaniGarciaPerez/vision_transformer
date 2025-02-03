import os
import torch
from torch import nn
import cv2
from pathlib import Path
from src.model_components.vit import VisionTransformer


D_MODEL = 768
N_CLASSES = 100
PATCH_SIZE = 16
N_HEADS = 3
MLP_SIZE = 3072
N_LAYERS = 12
DROPOUT_RATIO = 0.1
EPOCHS = 1
N_CHANNELS = 3
BATCH_SIZE = 10


images_path = os.path.join(Path(__file__).parent.parent.parent, "data", "test")

image_path = os.path.join(
    Path(__file__).parent.parent.parent, "data", "test", "cat.jpg"
)


images_batch = torch.stack(
    [
        torch.from_numpy(cv2.imread(image_path)).to(torch.float32).movedim(2, 0)
        for image in range(0, BATCH_SIZE)
    ],
    dim=0,
)

print(images_batch.shape)


model = VisionTransformer(
    PATCH_SIZE,
    D_MODEL,
    MLP_SIZE,
    N_HEADS,
    DROPOUT_RATIO,
    N_LAYERS,
    N_CLASSES,
    N_CHANNELS,
    BATCH_SIZE,
)

print(model(images_batch).shape)
