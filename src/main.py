"""
Author: Dani Garcia

Description: TODO: Insert

"""
import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from src.model_components.vit import VisionTransformer
from src.training_utils.dataset_loader import LoadDataset
from src.training_utils.train import TrainModel
from src.training_utils.optimizer import LoadOptimizer




if __name__=="__main__":
    load_dotenv(os.path.join(Path(__file__).parent.parent, "parameters.env"))

    D_MODEL = int(os.getenv("D_MODEL"))
    N_CLASSES = int(os.getenv("N_CLASSES"))
    PATCH_SIZE = int(os.getenv("PATCH_SIZE"))
    N_HEADS = int(os.getenv("N_HEADS"))
    MLP_SIZE = int(os.getenv("MLP_SIZE"))
    N_LAYERS = int(os.getenv("N_LAYERS"))
    DROPOUT_RATIO = float(os.getenv("DROPOUT_RATIO"))
    EPOCHS = int(os.getenv("EPOCHS"))
    N_CHANNELS = int(os.getenv("N_CHANNELS"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
    DATASET = os.getenv("DATASET", "FashionMnist")
    OPTIMIZER = os.getenv("OPTIMIZER", "ADAM")

    # Load model and insert training parameters
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

    # Define loss for multiclass classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = LoadOptimizer(model.parameters(), OPTIMIZER, LEARNING_RATE).get_optimizer()

    # Load training and validation datasets
    training_loader, validation_loader = LoadDataset(dataset_name=DATASET).load_data(BATCH_SIZE)

    # Train the model
    model_trainer = TrainModel(model, loss_fn, optimizer, EPOCHS)
    model_trainer.train(training_loader, validation_loader)
