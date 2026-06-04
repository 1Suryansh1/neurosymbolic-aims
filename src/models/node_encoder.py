import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15

class ODST(nn.Module):
    """
    Oblivious Decision State Tree (ODST)
    Uses entmax15 for sparse feature selection and routing decisions.
    """
    def __init__(self, in_features, depth=3):
        super().__init__()
        self.depth = depth
        
        # Differentiable sparse feature selection
        # Shape: [depth, in_features]
        self.feature_selector = nn.Parameter(torch.Tensor(depth, in_features))
        nn.init.uniform_(self.feature_selector, -0.1, 0.1)
        
        # Thresholds and scales for binary decisions
        self.thresholds = nn.Parameter(torch.Tensor(depth))
        self.scales = nn.Parameter(torch.ones(depth))
        nn.init.uniform_(self.thresholds, -1.0, 1.0)
        
        # Leaf values: 2^depth leaves
        self.num_leaves = 2 ** depth
        self.leaf_values = nn.Parameter(torch.randn(self.num_leaves))
        
        # Build binary codes for leaves to vectorize path probability computation
        # Shape: [num_leaves, depth]
        codes = []
        for i in range(self.num_leaves):
            binary = format(i, f'0{depth}b')
            codes.append([int(b) for b in binary])
        self.register_buffer('codes', torch.tensor(codes, dtype=torch.float32))

    def forward(self, x):
        # x: [batch, in_features]
        
        # 1. Sparse Feature Selection
        # feature_weights: [depth, in_features] (Sparse probabilities summing to 1)
        feature_weights = entmax15(self.feature_selector, dim=-1)
        
        # selected_features: [batch, depth]
        selected_features = F.linear(x, feature_weights)
        
        # 2. Compute Routing Logits
        # split_logits: [batch, depth]
        split_logits = (selected_features - self.thresholds) * self.scales
        
        # 3. Sparse Binary Routing Decisions
        # choices: [batch, depth, 2]
        choices = torch.stack([split_logits, -split_logits], dim=-1)
        p_choices = entmax15(choices, dim=-1)
        
        # p_right corresponds to choice 0 (split_logits), p_left to choice 1 (-split_logits)
        p_right = p_choices[:, :, 0] # [batch, depth]
        p_left = p_choices[:, :, 1]  # [batch, depth]
        
        p_right = p_right.unsqueeze(1) # [batch, 1, depth]
        p_left = p_left.unsqueeze(1)   # [batch, 1, depth]
        
        codes_expanded = self.codes.unsqueeze(0) # [1, num_leaves, depth]
        
        # Path probabilities: [batch, num_leaves, depth]
        paths = p_right * codes_expanded + p_left * (1 - codes_expanded)
        
        # Joint probability of reaching each leaf: [batch, num_leaves]
        leaf_probs = torch.prod(paths, dim=2)
        
        # Expected leaf value: [batch, 1]
        out = torch.matmul(leaf_probs, self.leaf_values).unsqueeze(-1)
        return out

class NODEEncoder(nn.Module):
    """
    Neural Oblivious Decision Ensemble (NODE) block.
    Acts as a direct replacement for an MLP on tabular data.
    """
    def __init__(self, input_dim, output_dim, num_trees=32, depth=3):
        super().__init__()
        # Ensemble of trees
        self.trees = nn.ModuleList([
            ODST(input_dim, depth=depth) for _ in range(num_trees)
        ])
        
        # Combine tree outputs and project to desired embedding dimension
        self.proj = nn.Sequential(
            nn.Linear(num_trees, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # Concatenate all tree outputs: [batch, num_trees]
        tree_outs = torch.cat([tree(x) for tree in self.trees], dim=1)
        # Project to target dimension (e.g. 64 for our metadata embedding)
        return self.proj(tree_outs)
