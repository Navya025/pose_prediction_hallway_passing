# Description: Contains the model for the Transformer network.

import torch
import torch.nn as nn
import math
from torch import Tensor

num_joints = 32
num_features = 7
num_frames = 5
max_seq_length = num_joints * num_features * num_frames

class CustomTransformerInput(nn.Module):
    """
    Input embedding module.
    """
    def __init__(self, d_model):
        super(CustomTransformerInput, self).__init__()
        self.embeddings = nn.Linear(num_joints * num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=num_frames)
        
    def forward(self, x):
        # Shape of x: (batch_size, num_frames, num_joints * num_features)
        embedded = self.embeddings(x)
        # Apply positional encoding to the embedded sequences
        return self.positional_encoding(embedded)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = num_frames):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``(batch_size, seq_len, embedding_dim)``
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class PosePredictionTransformer(nn.Module):
    """
    Main model for predicting joint orientations using nn.Transformer.
    """
    def __init__(self, num_layers, d_model, num_heads, num_features=7, seq_len=num_frames):
        super(PosePredictionTransformer, self).__init__()
        
        self.input_embedding = CustomTransformerInput(d_model)
        self.seq_len = seq_len

        self.transformer = nn.Transformer(d_model, num_heads, num_layers)
        
        # Output a num_framesx(32x7) matrix, using a time-distributed-like linear layer
        self.output_layer = nn.Linear(d_model, num_joints*num_features)

    def forward(self, x, mask=None):
        x = self.input_embedding(x)
        x = self.transformer.encoder(x)
        outputs = []
        # Loop through each token in the sequence
        for i in range(self.seq_len):
            out = self.output_layer(x[:, i, :])
            outputs.append(out)
        # Stack the outputs to get a [batch_size, seq_len, num_joints*num_features] tensor
        x = torch.stack(outputs, dim=1)
        return x

