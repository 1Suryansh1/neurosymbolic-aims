import torch
import torch.nn as nn

def get_prediction_loss(loss_type='huber', delta=1.0):
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'huber':
        return nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
