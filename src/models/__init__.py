import torch.nn as nn
from TSFEDL import OhShuLih, OhShuLih_Forecaster
from .dnn import TabularDNN
from .cnn import CNN
from .transformer import TabularTransformer
from .lstm import MultivariateLSTM
from .dlinear_adapter import DLinearAdapter
from .xlstm_adapter import xLSTMAdapter

class BackboneWrapper(nn.Module):
    def __init__(self, model, output_dim, permute_input=False, slice_method='end'):
        super().__init__()
        self.model = model
        self.output_dim = output_dim
        self.permute_input = permute_input
        self.slice_method = slice_method

    def forward(self, x):
        # Flatten parameters for RNNs
        if hasattr(self.model, "modules"):
            for m in self.model.modules():
                if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                    m.flatten_parameters()
        
        # Permute input if requested (N, L, C) -> (N, C, L)
        if self.permute_input:
             x = x.permute(0, 2, 1)
        
        out = self.model(x)
        
        # Check output dimension
        # Expect (N, L, C) structure for output usually
        if out.dim() == 3:
            # If output features > desired output_dim, slice
            # Assuming target variables are at the END of the vector
            if out.shape[2] > self.output_dim:
                if self.slice_method == 'start':
                    # Target variables at the BEGINNING (e.g. main_ts.py strategy)
                    out = out[:, :, :self.output_dim]
                else:
                    # Target variables at the END
                    out = out[:, :, -self.output_dim:]
                 
        return out

def get_model(model_cfg, input_dim, output_dim=1, device="cpu", **kwargs):
    # Support for consolidated config (backbone_tab.yaml)
    # If model_cfg has a nested config matching its 'name', use that sub-config combined with
    # top-level

    name = model_cfg.get("name", "dnn").lower()

    # Check if we have specific config for this model inside model_cfg (nested structure)
    if name in model_cfg:
        # Merge specific props into a new dict, preserving top-level props if needed or specific
        # overrides. Assuming specific config should take precedence for model params
        specific_cfg = model_cfg[name]
        # We might want to access properties from specific_cfg mostly
    else:
        # Fallback for flat config or if key missing
        specific_cfg = model_cfg

    if name == "dnn":
        # Extract seq_len from kwargs if available, default to 1
        seq_len = kwargs.get("seq_len", 1) 
        # Alternatively, check model_cfg
        if seq_len == 1:
            seq_len = model_cfg.get("window_size", 1)

        model = TabularDNN(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            hidden_dims=list(specific_cfg.get("hidden_dims", [128, 64, 32])),
            dropout_rate=specific_cfg.get("dropout_rate", 0.0)
        )
    elif name == "cnn":
        # CNN handles reshaping internally for (N, D) input if needed
        model = CNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=specific_cfg.get("hidden_dim", 64),
            kernel_size=specific_cfg.get("kernel_size", 3),
            dropout=specific_cfg.get("dropout", 0.2)
        )
    elif name == "transformer":
        # Uses FT-Transformer as a differentiable SOTA alternative for Backbone
        from .ft_transformer import FTTransformerBackbone
        model = FTTransformerBackbone(
            input_dim=input_dim,
            output_dim=output_dim,
            n_blocks=specific_cfg.get("n_blocks", 2),
            d_token=specific_cfg.get("d_token", 192),
            attention_dropout=specific_cfg.get("attention_dropout", 0.2),
            ffn_dropout=specific_cfg.get("ffn_dropout", 0.1),
            attention_n_heads=specific_cfg.get("attention_n_heads", 8),
        )

    # elif name == "transformer":
    #     model = TabularTransformer(
    #         input_dim=input_dim,
    #         output_dim=output_dim,
    #         d_model=model_cfg.get("d_model", 64),
    #         nhead=model_cfg.get("nhead", 4),
    #         num_layers=model_cfg.get("num_layers", 2),
    #         dim_feedforward=model_cfg.get("dim_feedforward", 128),
    #         dropout=model_cfg.get("dropout", 0.1)
    #     )
    elif name == "lstm":
        model = MultivariateLSTM(
            input_size=input_dim,
            output_size=output_dim,
            hidden_size=model_cfg.get("hidden_size", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2)
        )
    elif name == "dlinear":
        seq_len = kwargs.get("seq_len", model_cfg.get("seq_len", 24))
        pred_len = kwargs.get("pred_len", model_cfg.get("pred_len", 1))

        model = DLinearAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            moving_avg=specific_cfg.get("moving_avg", 25),
            individual=specific_cfg.get("individual", False)
        )

        # DLinearAdapter expects NLC natively.
        # Check if config requested Channels First (NCL). If so, we must permute NCL -> NLC.
        config_format = specific_cfg.get("format", "channels_last")
        permute = config_format == "channels_first"

        # DLinearAdapter ignores output_dim constructor arg for channel count,
        # so we must slice output to match target
        # For DLinear in TS context, we placed target at the BEGINNING of input
        model = BackboneWrapper(model, output_dim=output_dim, permute_input=permute, slice_method='start')

    elif name == "xlstm":
        seq_len = kwargs.get("seq_len", 24)
        model = xLSTMAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            hidden_dim=specific_cfg.get("hidden_size", specific_cfg.get("hidden_dim", 64)),
            num_layers=specific_cfg.get("num_layers", 2),
            dropout=specific_cfg.get("dropout", 0.1)
        )
    elif name == "ohshulih":
        # OhShuLih construction as in benchmark
        top_module = OhShuLih_Forecaster(
            out_features=output_dim,
            n_pred=1
        )
        model_base = OhShuLih(
            in_features=input_dim,
            top_module=top_module,
            loss=nn.MSELoss(),
        )

        # OhShuLih expects NCL natively.
        # We now standardize on NLC (Channels Last) input for all wrappers.
        # Therefore, we MUST permute NLC -> NCL.
        permute = True

        # Ensure correct input format and output dimension compatibility
        model = BackboneWrapper(model_base, output_dim=output_dim, permute_input=permute)

    else:
        raise ValueError(f"Model {name} not supported.")

    # Only call .to(device) if it's a PyTorch Module or explicitly supported and expected
    # to return self
    if isinstance(model, nn.Module):
        return model.to(device)

    # For TabPFN or others, assume they handle device internally or don't support .to() chaining
    return model
