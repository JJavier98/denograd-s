"""
This module implements a Multivariate Long Short-Term Memory (LSTM) network using PyTorch.
The module provides the `MultivariateLSTM` class, which is designed for time series forecasting
tasks where multiple input features are used to predict one or more target variables. It leverages
stacked LSTM layers to capture temporal dependencies in sequential data, followed by a linear
layer to project the final hidden state to the desired output dimension.
Classes:
    MultivariateLSTM: A PyTorch Module implementing an LSTM-based architecture for multivariate
        time series regression.
"""
import torch
import torch.nn as nn

class MultivariateLSTM(nn.Module):
    """
    Multivariate LSTM model for time series forecasting.
    This model utilizes Long Short-Term Memory (LSTM) layers to process multivariate 
    time series data. It takes a sequence of input features and predicts a target 
    vector based on the last hidden state of the sequence.
        input_size (int): Number of input variables (features) per time step.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers stacked.
        output_size (int): Number of variables to predict.
        dropout (float, optional): Dropout probability for LSTM layers if num_layers > 1. 
            Defaults to 0.2.
    Attributes:
        hidden_size (int): Stored size of the hidden state.
        num_layers (int): Stored number of recurrent layers.
        lstm (nn.LSTM): The LSTM layer processing the input sequence.
        fc (nn.Linear): Fully connected layer mapping the final hidden state to the output size.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Modelo LSTM para predicción multivariable de series temporales.

        Args:
            input_size (int): Número de variables de entrada (features) por paso de tiempo.
            hidden_size (int): Número de características en el estado oculto.
            num_layers (int): Número de capas recurrentes apiladas.
            output_size (int): Número de variables a predecir.
            dropout (float): Probabilidad de dropout si num_layers > 1.
        """
        super(MultivariateLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Definición de la capa LSTM
        # batch_first=True espera tensores de entrada con forma (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Capa totalmente conectada para mapear el estado oculto a la salida
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor de entrada con forma (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Predicción del último paso de tiempo con forma (batch_size, output_size)
        """
        # Inicializar estados ocultos y de celda (opcional, PyTorch lo hace a 0 por defecto)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Compactar pesos para evitar advertencias de uso de memoria y asegurar eficiencia en GPU
        self.lstm.flatten_parameters()

        # Propagación hacia adelante a través de LSTM
        # out tiene forma (batch_size, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Tomamos solo la salida del último paso de la secuencia
        # out[:, -1, :] tiene forma (batch_size, hidden_size)
        last_time_step = out[:, -1, :]

        # Pasamos por la capa lineal final
        prediction = self.fc(last_time_step)

        return prediction
