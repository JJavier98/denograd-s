"""Training utilities with early stopping and checkpoint persistence."""

import copy
import os

import numpy as np
import torch
from tqdm import tqdm

from src.utils import Colors, make_dir

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


class Trainer:
    """Small training helper with early stopping for benchmark experiments."""

    def __init__(
        self,
        model,
        train_generator,
        val_generator,
        device,
        criterion,
        optimizer,
        epoch_scheduler,
        batch_scheduler,
        patience,
        epochs,
        checkpoints_path,
        verbose=False,
        sparsity_controller=None,
    ) -> None:
        """Initialize the trainer and state used during optimization."""
        self.model = model
        self.best_model = copy.deepcopy(self.model)
        self.train_dataloader = train_generator
        self.val_dataloader = val_generator
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch_scheduler = epoch_scheduler
        self.batch_scheduler = batch_scheduler
        self.max_epochs = epochs
        self.best_val_loss = float("inf")
        self.train_loss_when_best_val_loss = np.inf
        self.patience = patience
        self.current_patience = 0
        self.checkpoints_path = checkpoints_path
        self.verbose = verbose
        self.sparsity_controller = sparsity_controller

    def __on_train_start(self):
        """Hook executed once before the first training epoch."""
        if self.sparsity_controller is not None:
            self.sparsity_controller.on_train_start(self.model, self.max_epochs)
        if self.verbose:
            print("Training is started!")

    def __on_train_epoch_start(self, epoch):
        """Set the wrapped model in training mode for a new epoch."""
        if self.sparsity_controller is not None:
            self.sparsity_controller.on_epoch_start(self.model, epoch, self.max_epochs)
        self.model.train(True)

    def __on_train_batch_start(self, batch_x, batch_y):
        """Run forward/backward/update for one training batch and return batch loss."""
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)

        self.optimizer.zero_grad()
        outputs = self.model(batch_x)

        if len(outputs.shape) < len(batch_y.shape):
            outputs = outputs.unsqueeze(1)

        loss = self.criterion(outputs, batch_y)
        loss.backward()
        self.optimizer.step()

        if self.sparsity_controller is not None:
            self.sparsity_controller.on_after_optimizer_step(self.model)

        if self.batch_scheduler is not None:
            self.batch_scheduler.step()

        return loss.item()

    def __on_val_start(self):
        """Switch the model to evaluation mode for validation."""
        self.model.train(False)
        self.model.eval()

    def __on_val_batch_start(self, batch_x_val, batch_y_val):
        """Compute validation loss for one batch."""
        batch_x_val = batch_x_val.to(self.device, dtype=torch.float32)
        batch_y_val = batch_y_val.to(self.device, dtype=torch.float32)

        val_outputs = self.model(batch_x_val)
        if len(val_outputs.shape) < len(batch_y_val.shape):
            val_outputs = val_outputs.unsqueeze(1)

        return self.criterion(val_outputs, batch_y_val).item()

    def __early_stopping(self, val_loss, train_loss, epoch):
        """Update early-stopping state and return True when training should stop."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.train_loss_when_best_val_loss = train_loss
            self.current_patience = 0
            self.best_model = copy.deepcopy(self.model)

            checkpoint_dir = os.path.dirname(self.checkpoints_path)
            if make_dir(checkpoint_dir):
                torch.save(self.model.state_dict(), self.checkpoints_path)

            if self.verbose:
                print(
                    f"\t{Colors.VERDE}Best model saved at epoch "
                    f"[{epoch + 1}/{self.max_epochs}]{Colors.RESET}"
                )
        else:
            self.current_patience += 1

        if self.current_patience >= self.patience:
            if self.verbose:
                print(
                    f"{Colors.ROJO}Training stopped after {Colors.AMARILLO}"
                    f"{self.patience}{Colors.ROJO} epochs without validation improvement."
                    f"{Colors.RESET}"
                )
            return True
        return False

    def fit(self):
        """Train the model and return best checkpoint plus tracked losses."""
        self.train_loss_when_best_val_loss = np.inf
        self.best_val_loss = np.inf
        train_losses = []
        val_losses = []
        self.__on_train_start()

        epochs_progress_bar = tqdm(
            range(self.max_epochs),
            total=self.max_epochs,
            leave=True,
            unit="epoch",
            desc="Epochs loop",
            colour="green",
        )

        for epoch in epochs_progress_bar:
            train_loss = 0.0
            val_loss = np.inf
            self.__on_train_epoch_start(epoch)

            batches_progress_bar = tqdm(
                self.train_dataloader,
                disable=not self.verbose,
                leave=False,
                unit="batch",
                desc="Batches loop",
            )
            for batch_x, batch_y in batches_progress_bar:
                train_loss += self.__on_train_batch_start(batch_x, batch_y) * len(batch_x)

                if self.verbose:
                    train_loss_str = train_loss / len(self.train_dataloader.dataset)
                    batches_progress_bar.set_postfix_str(
                        f"Train loss: {train_loss_str:.4f} - Val loss: {val_losses[-1]:.4f}"
                        if val_losses
                        else f"Train loss: {train_loss_str:.4f} - Val loss: N/A"
                    )

            train_loss /= len(self.train_dataloader.dataset)
            train_losses.append(train_loss)

            if self.val_dataloader is not None:
                val_loss = 0.0
                self.__on_val_start()
                with torch.no_grad():
                    for batch_x_val, batch_y_val in self.val_dataloader:
                        val_loss += self.__on_val_batch_start(batch_x_val, batch_y_val) * len(batch_x_val)
                val_loss /= len(self.val_dataloader.dataset)
                val_losses.append(val_loss)

            if wandb is not None and wandb.run is not None:
                log_dict = {
                    "backbone/train_loss": train_losses[-1],
                    "backbone/epoch": epoch,
                }
                if self.val_dataloader is not None and val_losses:
                    log_dict["backbone/val_loss"] = val_losses[-1]
                if self.sparsity_controller is not None:
                    sparsity_state = self.sparsity_controller.report()
                    for key, value in sparsity_state.items():
                        if isinstance(value, (int, float, bool)):
                            log_dict[f"backbone/sparsity/{key}"] = value
                wandb.log(log_dict)

            epochs_progress_bar.set_postfix_str(
                f"Train loss: {train_losses[-1]:.4f} - Val loss: {val_losses[-1]:.4f}"
                if train_losses and val_losses
                else f"Train loss: {train_losses[-1]:.4f}"
                if train_losses
                else "Train loss: N/A - Val loss: N/A"
            )

            tracked_val_loss = val_loss if self.val_dataloader is not None else train_loss
            if self.__early_stopping(tracked_val_loss, train_loss, epoch):
                break

            if self.epoch_scheduler is not None:
                try:
                    self.epoch_scheduler.step()
                except (RuntimeError, ValueError, TypeError):
                    self.epoch_scheduler.step(tracked_val_loss)

        return (
            self.best_model,
            train_losses,
            val_losses,
            self.train_loss_when_best_val_loss,
            self.best_val_loss,
        )

    def eval_dataloader(self, data_generator):
        """Run inference with the best model over a dataloader and return predictions."""
        predictions = []
        self.best_model.eval()
        for batch_x, _ in data_generator:
            batch_x = batch_x.to(self.device, dtype=torch.float32)
            predictions.append(self.best_model(batch_x).cpu().detach().numpy())
        return predictions