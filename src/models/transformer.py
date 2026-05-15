import torch
import torch.nn as nn

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        """
        Transformer model adapted for tabular data regression.
        Uses a Feature Tokenizer approach where each feature is embedded into a vector.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Feature Tokenizer using Grouped Conv1d for efficiency
        # Transforms each scalar feature x_i into a vector of size d_model
        # using independent weights for each feature.
        # Input: (Batch, Input_Dim, 1)
        # Output: (Batch, Input_Dim * d_model, 1) -> reshape -> (Batch, Input_Dim, d_model)
        self.embedding = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim * d_model,
            kernel_size=1,
            groups=input_dim
        )

        # [CLS] Token to aggregate information
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Head for regression
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Input_Dim)
        B, F = x.shape

        # --- Feature Tokenization ---
        # Reshape for Conv1d: (Batch, Channels, Length) -> (Batch, F, 1)
        x = x.unsqueeze(-1)

        # Apply embedding: (Batch, F*D, 1)
        x = self.embedding(x)

        # Reshape back to (Batch, F, D)
        x = x.view(B, F, self.d_model)

        # --- Add CLS Token ---
        # Expand CLS token to batch size: (Batch, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concatenate CLS token to the beginning of the sequence: (Batch, F + 1, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # --- Transformer Encoder ---
        # Output: (Batch, F + 1, D)
        x = self.transformer_encoder(x)

        # --- Readout ---
        # Take the CLS token output (index 0) corresponding to the aggregated representation
        x = x[:, 0, :]

        # --- Prediction ---
        x = self.mlp_head(x)

        return x
