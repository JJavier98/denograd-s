"""Dataset and dataloader helpers for tabular and time-series experiments."""

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset


class SlidingWindowDataset(Dataset):
    """Time series dataset with sliding-window indexing."""

    def __init__(self, X, Y, window_size, future, mode="discrete", cnn=False, stride=1):
        """Initialize sliding-window indexing over feature/target arrays."""
        self.X = X
        self.Y = Y
        self.window_size = window_size
        self.future = future
        self.mode = mode
        self.is_cnn = cnn
        self.stride = stride

    def __len__(self):
        """Return the number of available windows under current configuration."""
        if self.mode == "range":
            last_possible_start = len(self.X) - self.window_size - self.future
        else:
            last_possible_start = len(self.X) - self.window_size - max(self.future)

        if last_possible_start < 0:
            return 0
        return (last_possible_start // self.stride) + 1

    def __getitem__(self, idx):
        """Return one windowed sample pair as float tensors."""
        start_idx = idx * self.stride
        x = self.X[start_idx : start_idx + self.window_size]
        if self.is_cnn:
            x = x.T

        if self.mode == "range":
            y = self.Y[start_idx + self.window_size : start_idx + self.window_size + self.future]
        else:
            indices = [start_idx + self.window_size + i_fut - 1 for i_fut in self.future]
            y = self.Y[indices]

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        return [x, y]

    def copy(self):
        """Return a deep copy of the dataset instance."""
        return copy.deepcopy(self)


def load_data(data_path, target_col="y"):
    """Load tabular data and split into features and target arrays."""
    if not data_path:
        raise ValueError("data_path must be provided.")

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)

    target_cols = [target_col] if isinstance(target_col, str) else target_col
    missing_cols = [col for col in target_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Target columns {missing_cols} not found in dataset.")

    # Keep only numeric features to avoid failures with date/string columns.
    features_df = df.drop(columns=target_cols).select_dtypes(include=[np.number]).copy()
    if features_df.shape[1] == 0:
        raise ValueError("No numeric feature columns available after preprocessing.")

    # Targets are coerced to numeric when possible.
    targets_df = df[target_cols].apply(pd.to_numeric, errors="coerce")

    # Remove rows with invalid targets and impute feature NaNs with column medians.
    valid_rows = ~targets_df.isna().any(axis=1)
    features_df = features_df.loc[valid_rows]
    targets_df = targets_df.loc[valid_rows]
    features_df = features_df.fillna(features_df.median(numeric_only=True))

    X = features_df.values
    y = targets_df.values
    return X, y


def add_gaussian_noise(X, noise_std=0.0):
    """Add i.i.d. Gaussian noise to an array when noise_std is positive."""
    if noise_std <= 0:
        return X
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise


def fix_noise_seed(seed=42):
    """Seed numpy RNG for reproducible noise generation."""
    np.random.seed(seed)


def get_dataloaders(X, y, batch_size, test_split=0.2, val_split=0.1):
    """Create train/validation/test dataloaders for tabular arrays."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )
    val_size_relative = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_relative, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        (train_loader, val_loader, test_loader),
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
    )


def get_ts_dataloaders(X, y, window_size, future, batch_size, val_split=0.1, test_split=0.2, cnn=False):
    """Create train/validation/test sliding-window dataloaders for time series."""
    total_len = len(X)
    test_len = int(total_len * test_split)
    val_len = int(total_len * val_split)
    train_len = total_len - test_len - val_len

    X_train = X[:train_len]
    y_train = y[:train_len]

    start_val = max(0, train_len - window_size)
    end_val = train_len + val_len
    X_val = X[start_val:end_val]
    y_val = y[start_val:end_val]

    start_test = max(0, train_len + val_len - window_size)
    X_test = X[start_test:]
    y_test = y[start_test:]

    mode = "range" if isinstance(future, int) else "discrete"

    train_dataset = SlidingWindowDataset(X_train, y_train, window_size, future, mode=mode, cnn=cnn)
    val_dataset = SlidingWindowDataset(X_val, y_val, window_size, future, mode=mode, cnn=cnn)
    test_dataset = SlidingWindowDataset(X_test, y_test, window_size, future, mode=mode, cnn=cnn)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader), train_dataset, val_dataset, test_dataset
