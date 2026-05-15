import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, latent_dim: int,
                 dropout_rate: float = 0.0):
        """
        AutoEncoder model for tabular data and time series.
        
        Args:
            input_dim (int): Number of input features.
            hidden_layers (list): List of integers defining the size of encoding layers. 
                                  Decoding layers will be the reverse of this.
            latent_dim (int): Size of the bottleneck (latent representation).
            dropout_rate (float): Dropout probability.
        """
        super().__init__()

        self.encoder_layers = []
        self.decoder_layers = []

        # --- Encoder Construction ---
        prev_dim = input_dim
        for h_dim in hidden_layers:
            self.encoder_layers.append(nn.Linear(prev_dim, h_dim))
            self.encoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # Bottleneck
        self.encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder_net = nn.Sequential(*self.encoder_layers)

        # --- Decoder Construction ---
        prev_dim = latent_dim
        # Reverse hidden layers for symmetry
        for h_dim in reversed(hidden_layers):
            self.decoder_layers.append(nn.Linear(prev_dim, h_dim))
            self.decoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # Output layer (reconstruction)
        self.decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder_net = nn.Sequential(*self.decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the latent representation."""
        return self.encoder_net(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs the input from the latent representation."""
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        For regression/prediction tasks, the loss is typically MSE between input x and output.
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction
