import torch
import torch.nn as nn

class SymbolicLossLayer(nn.Module):
    def __init__(self, lambda_add=1.0, lambda_gdm=0.2, lambda_nonneg=0.1, log_transform=False):
        super().__init__()
        self.lambda_add = lambda_add
        self.lambda_gdm = lambda_gdm
        self.lambda_nonneg = lambda_nonneg
        self.log_transform = log_transform
        
    def forward(self, predictions):
        # predictions shape: [batch, 5]
        # Target index map:
        # 0: Dry_Clover_g
        # 1: Dry_Dead_g
        # 2: Dry_Green_g
        # 3: Dry_Total_g
        # 4: GDM_g
        
        if self.log_transform:
            # We reconstruct raw scale values to compute the sum, but we compute losses in log space
            clover_raw = torch.expm1(torch.clamp(predictions[:, 0], min=-1.0, max=10.0))
            dead_raw = torch.expm1(torch.clamp(predictions[:, 1], min=-1.0, max=10.0))
            green_raw = torch.expm1(torch.clamp(predictions[:, 2], min=-1.0, max=10.0))
            
            # Reconstructed raw total
            calc_total_raw = clover_raw + dead_raw + green_raw
            # Map back to log space
            calc_total_log = torch.log1p(torch.clamp(calc_total_raw, min=0.0))
            
            # 1. Additive Rule: MSE in log space
            loss_add = torch.mean((predictions[:, 3] - calc_total_log) ** 2)
            
            # 2. GDM-Green proximity rule: MSE in log space
            loss_gdm = torch.mean((predictions[:, 4] - predictions[:, 2]) ** 2)
            
            # 3. Non-negativity rule: penalty on raw scale clamped (keep on raw scale to push away from negatives)
            loss_nonneg = torch.mean(torch.clamp(-torch.expm1(predictions), min=0.0) ** 2)
        else:
            clover = predictions[:, 0]
            dead = predictions[:, 1]
            green = predictions[:, 2]
            total = predictions[:, 3]
            gdm = predictions[:, 4]
            
            # 1. Additive Rule: Total = Clover + Dead + Green
            calc_total = clover + dead + green
            loss_add = torch.mean((total - calc_total) ** 2)
            
            # 2. GDM-Green proximity rule: GDM ≈ Green
            loss_gdm = torch.mean((gdm - green) ** 2)
            
            # 3. Non-negativity rule: all values must be >= 0
            loss_nonneg = torch.mean(torch.clamp(-predictions, min=0.0) ** 2)
            
        # Total symbolic loss
        total_symbolic_loss = (
            self.lambda_add * loss_add +
            self.lambda_gdm * loss_gdm +
            self.lambda_nonneg * loss_nonneg
        )
        
        return {
            'loss_symbolic': total_symbolic_loss,
            'loss_add': loss_add,
            'loss_gdm': loss_gdm,
            'loss_nonneg': loss_nonneg
        }
