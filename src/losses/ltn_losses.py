import torch
import torch.nn as nn
import ltn

class AdditivityFunc(nn.Module):
    def forward(self, preds, *args, **kwargs):
        # Additivity — how "true" is Total ≈ Clover+Dead+Green?
        truth = 1 - torch.abs(preds[:,3] - (preds[:,0]+preds[:,1]+preds[:,2])) / (torch.abs(preds[:,3]) + 1e-6)
        return torch.clamp(truth, min=0.0, max=1.0)

class ProximityFunc(nn.Module):
    def forward(self, preds, *args, **kwargs):
        # Proximity — GDM ≈ Green
        truth = 1 - torch.abs(preds[:,4] - preds[:,2]) / (torch.abs(preds[:,2]) + 1e-6)
        return torch.clamp(truth, min=0.0, max=1.0)

class NonNegFunc(nn.Module):
    def forward(self, preds, *args, **kwargs):
        # Non-negativity — all outputs >= 0
        return torch.sigmoid(preds.min(dim=1).values)


class LTNLossLayer(nn.Module):
    def __init__(self, lambda_ltn=1.0, is_log_space=False):
        """
        Neuro-Symbolic V3 Loss layer using LTNtorch.
        """
        super().__init__()
        self.lambda_ltn = lambda_ltn

        # Define universal quantifier
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        self.sat_agg = ltn.fuzzy_ops.SatAgg()

        # Wrap predicates in nn.Module as required by LTNtorch
        self.Additivity = ltn.Predicate(AdditivityFunc())
        self.Proximity = ltn.Predicate(ProximityFunc())
        self.NonNeg = ltn.Predicate(NonNegFunc())

    def forward(self, preds, pred_loss):
        """
        preds: [batch_size, 5]
        pred_loss: scalar tensor of the standard predictive loss (Huber/MSE)
        """
        # We need an LTN variable to quantify over.
        x = ltn.Variable('x', preds)

        # Aggregate satisfaction
        sat = self.sat_agg(
            self.Forall(x, self.Additivity(x)),
            self.Forall(x, self.Proximity(x)),
            self.Forall(x, self.NonNeg(x))
        )

        # Loss is 1 - sat
        ltn_loss = 1.0 - sat
        
        # Combine with prediction loss
        total_loss = pred_loss + self.lambda_ltn * ltn_loss
        
        return total_loss, ltn_loss
