# Description: Contains the model for the Transformer network.

import torch
import torch.nn as nn
import math

num_joints = 31
num_features = 7
num_frames = 5
max_seq_length = num_joints * num_features * num_frames

class CustomTransformerInput(nn.Module):
    """
    Input embedding module.
    """
    def __init__(self, d_model):
        super(CustomTransformerInput, self).__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
    def forward(self, x):
        features = torch.split(x, 1, dim=-1)
        embedded_features = [embed(feature) for embed, feature in zip(self.embeddings, features)]
        combined_embedding = sum(embedded_features)
        # Apply positional encoding to the combined embeddings
        return self.positional_encoding(combined_embedding)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class PosePredictionTransformer(nn.Module):
    """
    Main model for predicting joint orientations using nn.Transformer.
    """
    def __init__(self, num_layers, d_model, num_heads, num_features=7):
        super(PosePredictionTransformer, self).__init__()
        
        self.input_embedding = CustomTransformerInput(d_model)

        self.transformer = nn.Transformer(d_model, num_heads, num_layers)
        
        # Output layer for waist joint pose prediction
        self.output_layer = nn.Linear(d_model, num_features)

    def forward(self, x, mask=None):
        x = self.input_embedding(x)
        x = self.transformer.encoder(x)
        # Taking the last position in the sequence, which should correspond to the waist joint
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x
