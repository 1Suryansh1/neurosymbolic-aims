import torch
import torch.nn as nn

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Handle batch of size 1 if needed (BatchNorm needs > 1 batch size during training,
        # but during evaluation it works fine). In PyTorch, we can also use LayerNorm.
        # But standard BatchNorm1d works fine with eval mode.
        return self.mlp(x)
