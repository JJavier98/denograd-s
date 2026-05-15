import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, kernel_size=3, dropout=0.2):
        super(CNN, self).__init__()

        # Determine padding to keep dimensions consistent where possible
        padding = kernel_size // 2

        self.features = nn.Sequential(
            # First Conv Block
            nn.Conv1d(
                in_channels=1,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            # Second Conv Block
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim * 2,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            # Third Conv Block
            nn.Conv1d(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim * 4,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Use Global Average Pooling to reduce dimensions and parameters
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output after pooling is (Batch, hidden_dim*4, 1), so flattened is just hidden_dim*4
        flattened_size = hidden_dim * 4

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Expecting input shape: (batch_size, input_dim)
        # Reshape for Conv1d: (batch_size, channels, sequence_length)
        # We treat tabular features as a sequence of length `input_dim` with 1 channel.
        x = x.unsqueeze(1)

        x = self.features(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x
