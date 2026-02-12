import torch
import torch.nn as nn

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Project down to low rank (V part of SVD)
        # shape: (rank, in_features)
        self.project_in = nn.Linear(in_features, rank, bias=False)
        
        # Project up to output dimension (U part of SVD)
        # shape: (out_features, rank)
        self.project_out = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.project_out(self.project_in(x))
    
    @classmethod
    def from_linear(cls, linear_layer, rank):
        """Creates a LowRankLinear layer from an existing nn.Linear layer."""
        return cls(linear_layer.in_features, linear_layer.out_features, rank, bias=linear_layer.bias is not None)
