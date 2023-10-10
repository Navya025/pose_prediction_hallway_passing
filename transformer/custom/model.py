# Description: Contains the model for the Transformer network.

import torch
import torch.nn as nn


class CustomTransformerInput(nn.Module):
    """
    Custom input embedding module.
    """
    def __init__(self, feature_size, d_model):
        super(CustomTransformerInput, self).__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(feature_size)])
        
    def forward(self, x):
        features = torch.split(x, 1, dim=-1)
        embedded_features = [embed(feature) for embed, feature in zip(self.embeddings, features)]
        combined_embedding = sum(embedded_features)
        return combined_embedding



class TransformerBlock(nn.Module):
    """
    A single transformer block with multi-head attention followed by a feed-forward network.
    """
    def __init__(self, d_model, num_heads):
        super(TransformerBlock, self).__init__()

        # MultiHead Attention
        self.attention = nn.MultiheadAttention(d_model, num_heads)

        # Feed-forward layers
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # expand dimensionality
            nn.ReLU(), #activation function
            nn.Linear(d_model * 4, d_model) # contract dimensionality to original size
        )

        # Layer Normalizations
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Attention layer
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(self.layer_norm1(attn_output))
        
        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(self.layer_norm2(ff_output))
        
        return x


class PosePredictionTransformer(nn.Module):
    """
    Main model for predicting joint orientations.
    """
    def __init__(self, num_layers, d_model, num_heads, feature_size=6):
        super(PosePredictionTransformer, self).__init__()
        
        # Custom input embedding
        self.input_embedding = CustomTransformerInput(feature_size, d_model)
        
        # Stack of Transformer blocks (hidden layers)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )

        # Output layer
        # Assuming the output also has 7 features for each joint
        # May need to update this to only output pelvis joint
        self.output_layer = nn.Linear(d_model, feature_size)

    def forward(self, x, mask=None):
        x = self.input_embedding(x)

        for block in self.blocks:
            x = block(x, mask=mask)
        
        x = self.output_layer(x)
        return x

# TODO: 
# NEED TO IMPLEMENT POSITIONAL ENCODING
# NEED TO ENSURE INPUT EMBEDDING LAYER IS SUITABLE FOR OUR USE CASE
# NEED TO CUSTOMIZE OUTPUT LAYER TO GET ONLY WAIST JOINT POSE
