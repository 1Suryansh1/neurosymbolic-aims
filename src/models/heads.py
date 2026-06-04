import torch
import torch.nn as nn

class MultiTargetHead(nn.Module):
    def __init__(self, input_dim=128, num_targets=5):
        super().__init__()
        
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Target heads: One scalar linear layer for each of the 5 targets
        # Targets: Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g
        self.heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_targets)
        ])
        
    def forward(self, x):
        shared_feat = self.shared(x)
        # Predict each target individually and concatenate
        outputs = [head(shared_feat) for head in self.heads]
        return torch.cat(outputs, dim=1) # Shape: [batch, num_targets]
