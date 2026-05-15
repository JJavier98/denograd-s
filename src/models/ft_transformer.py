import torch
import torch.nn as nn
import rtdl

class FTTransformerBackbone(nn.Module):
    """
    Adapter for the FT-Transformer model from the rtdl library.
    FT-Transformer is a SOTA Transformer architecture for tabular data.
    """
    def __init__(self, input_dim, output_dim, n_blocks=2, d_token=192, attention_dropout=0.2, ffn_dropout=0.1, attention_n_heads=8, ffn_d_hidden=None):
        super().__init__()

        # FT-Transformer expects categorical and numerical features separately.
        # Since we are likely dealing with already preprocessed/numerical data (or simple float tensors),
        # we will treat all 'input_dim' features as numerical.

        # 2. Transformer Backbone constructed manually to support all parameters
        #    and avoid 'make_default' restrictions.
        if ffn_d_hidden is None:
             # Best practice for ReGLU: 4/3 * d_token (approximation for GLU parameters)
            ffn_d_hidden = int(d_token * 4 / 3)

        transformer = rtdl.Transformer(
            d_token=d_token,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            attention_initialization='kaiming',
            attention_normalization='LayerNorm',
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=ffn_dropout,
            ffn_activation='ReGLU',
            ffn_normalization='LayerNorm',
            residual_dropout=0.0,
            prenormalization=True,
            first_prenormalization=False,
            last_layer_query_idx=None,
            n_tokens=None,
            kv_compression_ratio=None,
            kv_compression_sharing=None,
            head_activation='ReLU',
            head_normalization='LayerNorm',
            d_out=output_dim,
        )

        # 1. Feature Tokenizer to project numerical inputs to d_token embeddings
        tokenizer = rtdl.FeatureTokenizer(
            n_num_features=input_dim,
            cat_cardinalities=[],  # No categorical features assumed
            d_token=d_token,
        )

        self.model = rtdl.FTTransformer(feature_tokenizer=tokenizer, transformer=transformer)

    def forward(self, x):
        # x shape: (N, D)
        # FT-Transformer (rtdl) expects separate arguments for x_num and x_cat
        # We only pass x_num.

        if x.dim() > 2:
            # Flatten (N, L, D) -> (N, L*D) or similar if needed.
            # Assuming DenoGrad tabular passes (N, D)
             x = x.view(x.size(0), -1)

        return self.model(x, None)
