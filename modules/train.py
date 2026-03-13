import gpytorch
import torch
from torch.utils.data import random_split

def train_valid_split(
        dataset: torch.utils.data.Dataset,
        train_fraction: float= 0.8,
        seed: int = 42,
    ):
    train_size = int(train_fraction * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, valid_dataset

def get_tensors_from_subset(subset):
    idx = subset.indices
    X = subset.dataset.X[idx]
    y = subset.dataset.y[idx]
    return X, y

import torch
import gpytorch
from copy import deepcopy


def train(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model: gpytorch.models.ExactGP,
    n_epochs: int = 100,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = "best_gpr_model.pt",
    metadata: dict | None = None,
):

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    valid_x = valid_x.to(device)
    valid_y = valid_y.to(device)

    history = {
        "loss": [],
        "mae_train": [],
        "mae_valid": [],
        "lr": [],
    }

    best_mae_valid = float("inf")
    best_epoch = -1

    for i in range(n_epochs):
        model.train()
        model.likelihood.train()

        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

        if (i + 1) % 100 == 0 or i == 0:

            model.eval()
            model.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_train = model.likelihood(model(train_x)).mean
                pred_valid = model.likelihood(model(valid_x)).mean

                mae_train = torch.mean(torch.abs(pred_train - train_y))
                mae_valid = torch.mean(torch.abs(pred_valid - valid_y))

            loss_value = loss.item()
            mae_train_value = mae_train.item()
            mae_valid_value = mae_valid.item()
            lr_value = optimizer.param_groups[0]["lr"]

            history["loss"].append(loss_value)
            history["mae_train"].append(mae_train_value)
            history["mae_valid"].append(mae_valid_value)
            history["lr"].append(lr_value)

            if mae_valid_value < best_mae_valid:
                best_mae_valid = mae_valid_value
                best_epoch = i + 1

                checkpoint = {
                    "epoch": best_epoch,
                    "best_mae_valid": best_mae_valid,
                    "model_state_dict": deepcopy(model.state_dict()),
                    "likelihood_state_dict": deepcopy(model.likelihood.state_dict()),
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                    "scheduler_state_dict": deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                    "train_x": train_x.detach().cpu(),
                    "train_y": train_y.detach().cpu(),
                    "valid_x": valid_x.detach().cpu(),
                    "valid_y": valid_y.detach().cpu(),
                    "history": history,
                    "metadata": metadata if metadata is not None else {},
                }
                torch.save(checkpoint, checkpoint_path)


            print(
                f"Iter {i+1}/{n_epochs} "
                f"Loss: {loss_value:.6f} "
                f"MAE train: {mae_train_value:.6f} "
                f"MAE valid: {mae_valid_value:.6f} "
                f"best MAE valid: {best_mae_valid:.6f} "
                f"noise: {model.likelihood.noise.item():.6f} "
                f"lr: {lr_value:.3e}"
            )

    print(f"Best validation MAE: {best_mae_valid:.6f} at epoch {best_epoch}")
    print(f"Best checkpoint saved to: {checkpoint_path}")

    return history, best_mae_valid