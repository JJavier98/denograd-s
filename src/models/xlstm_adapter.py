import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig

class xLSTMAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()

        # xLSTM configuration
        # Assuming a mix of mLSTM and sLSTM or just one type.
        # Using a simple default configuration.

        # Note: xLSTM implementations might require embedding_dim (d_model).
        # We project input_dim to hidden_dim (embedding_dim) first if needed,
        # or set embedding_dim = input_dim if supported.
        # xLSTM usually works on embeddings.

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
            slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4)), # vanilla backend for compatibility
            context_length=seq_len,
            embedding_dim=hidden_dim,
            num_blocks=num_layers,
            dropout=dropout
        )

        self.xlstm = xLSTMBlockStack(cfg)

        # Head for regression
        # xLSTM outputs (Batch, Seq, Hidden)
        # We take the last time step or average? Usually last for forecasting.
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (Batch, Seq, InputDim)

        # Project to embedding dim
        x = self.input_proj(x)

        # xLSTM
        x = self.xlstm(x)

        # Take last time step
        # x shape: (Batch, Seq, Hidden)
        x_last = x[:, -1, :]

        out = self.head(x_last)
        return out
