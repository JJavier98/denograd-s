"""
This module defines a simple Deep Neural Network (DNN) architecture tailored for tabular data
and flattened time series regression tasks using PyTorch. The `TabularDNN` class constructs
a fully connected feed-forward neural network with configurable depth, width, dropout
regularization, and activation functions. It supports standard tabular inputs as well as
time series inputs by flattening the sequence dimension.

Classes:
    TabularDNN: A PyTorch module representing a configurable Multi-Layer Perceptron (MLP).
Dependencies:
    torch
    torch.nn
"""
import torch
import torch.nn as nn

class TabularDNN(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) for tabular and time series regression.

    Attributes:
        seq_len (int): Length of the input sequence.
        model (nn.Sequential): The sequential container of layers that form the network.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        seq_len: int = 1,
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.0,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initializes the TabularDNN model.
        
        Args:
            input_dim (int): Number of input features per time step (or per sample for
                tabular data).
            output_dim (int, optional): Number of output neurons. Defaults to 1.
            seq_len (int, optional): Length of the input sequence. Defaults to 1 (tabular).
                If  > 1, inputs are flattened from (Batch, Seq, Feat) to (Batch, Seq*Feat).
            hidden_dims (list, optional): List of integers containing the size of each hidden layer.
                Defaults to [128, 64, 32].
            dropout_rate (float, optional): Probability of dropout. Defaults to 0.0.
            activation (nn.Module, optional): Activation function to use in hidden layers.
                Defaults to nn.ReLU().
        """
        super().__init__()

        layers = []
        self.seq_len = seq_len
        in_features = input_dim * seq_len

        for dim in hidden_dims:
            layers.append(nn.Linear(in_features, dim))
            if activation:
                layers.append(activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = dim

        # Output layer for regression (output_dim neurons, no activation)
        layers.append(nn.Linear(in_features, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation performed at every call. Handles both 2D tabular data and 
        3D time series data by flattening the input if necessary.

        Args:
            x (torch.Tensor): The input tensor to the model. 
                Shape can be (batch_size, input_dim * seq_len) or (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The output tensor resulting from passing the input through the model
                layers. Shape: (batch_size, output_dim).
        """
        if x.dim() == 3:
            # Flatten input if it comes from a time series dataset (Batch, Seq, Features)
            x = x.reshape(x.size(0), -1)

        return self.model(x)
