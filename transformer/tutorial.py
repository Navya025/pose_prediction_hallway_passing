import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class PosePredictionTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_joints=22, data_size=7, num_frames=5):
        super(PosePredictionTransformer, self).__init__()
        
        # Assuming each joint data is embedded as a vector of size d_model
        self.input_embedding = nn.Linear(num_joints * data_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, data_size * num_joints * num_frames)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff=4*d_model, dropout=0.1) for _ in range(num_layers)])
        
        # This will convert the sequence to a single vector representation
        self.fc1 = nn.Linear(num_joints * d_model, d_model)
        
        # Predict the future waist data
        self.output_layer = nn.Linear(d_model, data_size)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, None)  # Assuming no mask for now

        x = x.view(x.size(0), -1)  # Flatten the sequence
        x = self.fc1(x)
        x = self.output_layer(x)
        return x
