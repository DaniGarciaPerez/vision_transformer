"""
Author: Dani Garcia

Description: TODO: Insert

"""

import torch
from src.model_components.vit import VisionTransformer
from src.training_utils.dataset_loader import LoadDataset
from src.training_utils.train import TrainModel

D_MODEL = 256
N_CLASSES = 100
PATCH_SIZE = 16
N_HEADS = 12
MLP_SIZE = 3072
N_LAYERS = 12
DROPOUT_RATIO = 0.1
EPOCHS = 3
N_CHANNELS = 3
BATCH_SIZE = 200
LEARNING_RATE = 0.001

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

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

training_loader, validation_loader = LoadDataset().load_data(BATCH_SIZE, "CIFAR100")

model_trainer = TrainModel(model, loss_fn, optimizer, EPOCHS)
model_trainer.train(training_loader, validation_loader)
