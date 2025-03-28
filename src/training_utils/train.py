"""
Author: Dani Garcia

Description: TODO: Insert

"""

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class TrainModel:
    def __init__(self, model: object, loss_fn: object, optimizer: object, epochs: int):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs

    def one_epoch(self, epoch_index, tb_writer, training_loader):
        """
        TODO: Insert docstring
        """
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(training_loader)):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 50 == 49:
                last_loss = running_loss / 50  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss

    def train(self, training_loader, validation_loader):
        """
        TODO: Insert docstring
        """

        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
        epoch_number = 0

        best_vloss = 1_000_000.0

        for epoch in range(self.epochs):

            print("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.one_epoch(epoch_number, writer, training_loader)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number + 1,
            )
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = "model_{}_{}".format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
