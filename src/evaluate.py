import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from src.config import Config

@torch.no_grad()
def evaluate_model(model, dataloader, device, log_transform=False):
    """
    Evaluates the PyTorch model on the dataloader and computes all prediction and constraint metrics.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for images, meta, targets in dataloader:
        images = images.to(device)
        meta = meta.to(device)
        
        preds, _ = model(images, meta)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
    all_preds = np.concatenate(all_preds, axis=0)      # Shape: [N, 5]
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: [N, 5]
    
    # Invert log transform if applied during training
    if log_transform:
        all_preds_raw = np.expm1(np.clip(all_preds, -1.0, 10.0))
        all_targets_raw = np.expm1(all_targets)
    else:
        all_preds_raw = all_preds
        all_targets_raw = all_targets
        
    # Standard Regression Metrics per Target
    metrics = {}
    targets_names = Config.TARGETS
    
    mae_list = []
    rmse_list = []
    
    for i, name in enumerate(targets_names):
        mae = mean_absolute_error(all_targets_raw[:, i], all_preds_raw[:, i])
        rmse = root_mean_squared_error(all_targets_raw[:, i], all_preds_raw[:, i])
        try:
            r2 = r2_score(all_targets_raw[:, i], all_preds_raw[:, i])
        except Exception:
            r2 = float('nan')
            
        mae_list.append(mae)
        rmse_list.append(rmse)
        
        metrics[f"{name}_MAE"] = mae
        metrics[f"{name}_RMSE"] = rmse
        metrics[f"{name}_R2"] = r2
        
    # Global metrics
    metrics['Mean_MAE'] = np.mean(mae_list)
    metrics['Mean_RMSE'] = np.mean(rmse_list)
    
    # Constraint Consistency Metrics (Computed on raw predictions)
    clover_pred = all_preds_raw[:, 0]
    dead_pred = all_preds_raw[:, 1]
    green_pred = all_preds_raw[:, 2]
    total_pred = all_preds_raw[:, 3]
    gdm_pred = all_preds_raw[:, 4]
    
    # Additive rule violation: Total - (Clover + Dead + Green)
    calc_total_pred = clover_pred + dead_pred + green_pred
    add_violation = np.abs(total_pred - calc_total_pred)
    
    metrics['Mean_Add_Violation'] = np.mean(add_violation)
    metrics['Max_Add_Violation'] = np.max(add_violation)
    metrics['Add_Violation_lt_0_1'] = np.mean(add_violation < 0.1)
    metrics['Add_Violation_lt_1_0'] = np.mean(add_violation < 1.0)
    metrics['Add_Violation_lt_5_0'] = np.mean(add_violation < 5.0)
    
    # GDM-Green discrepancy: GDM - Green
    gdm_green_disc = np.abs(gdm_pred - green_pred)
    metrics['Mean_GDM_Green_Discrepancy'] = np.mean(gdm_green_disc)
    
    # Negative predictions fraction
    metrics['Fraction_Negative_Predictions'] = np.mean(all_preds_raw < 0)
    
    # Return predictions, targets and all metrics
    return all_preds_raw, all_targets_raw, metrics
