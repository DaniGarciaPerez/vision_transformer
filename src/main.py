"""
Author: Dani Garcia

Description: 
    Main entry point for training a Vision Transformer (ViT) model on a specified dataset.

Usage:
    python main.py
"""

import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from src.model_components.vit import VisionTransformer
from src.training_utils.dataset_loader import LoadDataset
from src.training_utils.train import TrainModel
from src.training_utils.optimizer import LoadOptimizer

if __name__ == "__main__":

    # Load environment variables from.env file
    load_dotenv(os.path.join(Path(__file__).parent.parent, "parameters.env"))

    # Model and training HP
    D_MODEL = int(os.getenv("D_MODEL"))
    N_CLASSES = int(os.getenv("N_CLASSES"))
    PATCH_SIZE = int(os.getenv("PATCH_SIZE"))
    N_HEADS = int(os.getenv("N_HEADS"))
    MLP_SIZE = int(os.getenv("MLP_SIZE"))
    HIDDEN_LAYER_SIZE = int(os.getenv("HIDDEN_LAYER_SIZE"))
    N_LAYERS = int(os.getenv("N_LAYERS"))
    DROPOUT_RATIO = float(os.getenv("DROPOUT_RATIO"))
    EPOCHS = int(os.getenv("EPOCHS"))
    N_CHANNELS = int(os.getenv("N_CHANNELS"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
    DATASET = os.getenv("DATASET", "FashionMnist")
    OPTIMIZER = os.getenv("OPTIMIZER", "ADAM")

    # Instantiate ViT
    model = VisionTransformer(
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        mlp_size=MLP_SIZE,
        hidden_class_layer=HIDDEN_LAYER_SIZE,
        n_heads=N_HEADS,
        dropout_ratio=DROPOUT_RATIO,
        n_layers=N_LAYERS,
        n_classes=N_CLASSES,
        n_channels=N_CHANNELS,
    )

    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Instantiate Optimizer
    optimizer = LoadOptimizer(
        model_parameters=model.parameters(),
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE
    ).get_optimizer()

    # Load Training and Validation Datasets
    training_loader, validation_loader = LoadDataset(dataset_name=DATASET).load_data(batch_size=BATCH_SIZE)

    # Train the Model
    model_trainer = TrainModel(model, loss_fn, optimizer, epochs=EPOCHS)
    model_trainer.train(training_loader, validation_loader)