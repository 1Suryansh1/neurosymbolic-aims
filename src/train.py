import os
import argparse
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from src.config import Config
from src.dataset import get_dataloaders
from src.models.multimodal_model import MultimodalModel
from src.losses.prediction_loss import get_prediction_loss
from src.losses.symbolic_losses import SymbolicLossLayer
from src.losses.ltn_losses import LTNLossLayer
from src.evaluate import evaluate_model

def train_one_epoch(model, dataloader, optimizer, pred_criterion, sym_criterion, device, neuro_symbolic=False, neuro_symbolic_v3=False):
    model.train()
    
    epoch_pred_loss = 0.0
    epoch_sym_loss = 0.0
    epoch_total_loss = 0.0
    
    for images, meta, targets in dataloader:
        images = images.to(device)
        meta = meta.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        preds, _ = model(images, meta)
        
        # 1. Prediction Loss
        loss_pred = pred_criterion(preds, targets)
        loss_total = loss_pred
        
        # 2. Symbolic Consistency Loss
        loss_sym_dict = {}
        if (neuro_symbolic or neuro_symbolic_v3) and sym_criterion is not None:
            if neuro_symbolic_v3:
                # LTNLossLayer expects (preds, pred_loss)
                loss_total, loss_sym = sym_criterion(preds, loss_pred)
            else:
                loss_sym_dict = sym_criterion(preds)
                loss_sym = loss_sym_dict['loss_symbolic']
                loss_total = loss_total + loss_sym
            epoch_sym_loss += loss_sym.item()
            
        loss_total.backward()
        optimizer.step()
        
        epoch_pred_loss += loss_pred.item()
        epoch_total_loss += loss_total.item()
        
    n_batches = len(dataloader)
    return {
        'loss_pred': epoch_pred_loss / n_batches,
        'loss_sym': epoch_sym_loss / n_batches if (neuro_symbolic or neuro_symbolic_v3) else 0.0,
        'loss_total': epoch_total_loss / n_batches
    }

def main():
    parser = argparse.ArgumentParser(description="Train Biomass Prediction Model")
    parser.add_argument('--fold', type=int, default=0, help='CV Fold to train')
    parser.add_argument('--use_image', action='store_true', default=False, help='Use image branch')
    parser.add_argument('--use_meta', action='store_true', default=False, help='Use metadata branch')
    parser.add_argument('--use_node', action='store_true', default=False, help='Use NODE for metadata branch')
    parser.add_argument('--neuro_symbolic', action='store_true', default=False, help='Enable symbolic rule loss')
    parser.add_argument('--neuro_symbolic_v3', action='store_true', default=False, help='Enable LTN V3 loss')
    parser.add_argument('--log_targets', action='store_true', default=False, help='Log1p transform targets')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=Config.LR, help='Learning rate')
    parser.add_argument('--lambda_add', type=float, default=Config.LAMBDA_ADD, help='Additivity constraint weight')
    parser.add_argument('--lambda_gdm', type=float, default=Config.LAMBDA_GDM, help='Proximity constraint weight')
    parser.add_argument('--lambda_nonneg', type=float, default=Config.LAMBDA_NONNEG, help='Nonnegativity constraint weight')
    parser.add_argument('--model_name', type=str, default=None, help='Custom model checkpoint name')
    parser.add_argument('--unfreeze_epoch', type=int, default=-1, help='Epoch to unfreeze last CNN block (-1 for never)')
    
    args = parser.parse_args()
    
    # If neither branch is selected, default to multimodal (both True)
    if not args.use_image and not args.use_meta:
        print("No branches selected. Defaulting to Multimodal (use_image=True, use_meta=True).")
        args.use_image = True
        args.use_meta = True
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load Dataloaders
    train_loader, val_loader, meta_dim = get_dataloaders(
        fold=args.fold, 
        batch_size=Config.BATCH_SIZE, 
        image_size=Config.IMAGE_SIZE, 
        log_transform_targets=args.log_targets,
        use_image=args.use_image
    )
    
    # Initialize Model
    model = MultimodalModel(
        meta_input_dim=meta_dim,
        image_emb_dim=Config.IMAGE_EMB_DIM,
        meta_emb_dim=Config.META_EMB_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_targets=Config.NUM_TARGETS,
        use_image=args.use_image,
        use_meta=args.use_meta,
        use_node=args.use_node,
        freeze_backbone=Config.FREEZE_BACKBONE,
        backbone_name=Config.BACKBONE
    ).to(device)
    
    # Setup Losses
    pred_criterion = get_prediction_loss(loss_type=Config.PRED_LOSS_TYPE)
    sym_criterion = None
    if args.neuro_symbolic_v3:
        sym_criterion = LTNLossLayer(
            lambda_ltn=args.lambda_add, # Reusing this config
            is_log_space=args.log_targets
        ).to(device)
    elif args.neuro_symbolic:
        sym_criterion = SymbolicLossLayer(
            lambda_add=args.lambda_add,
            lambda_gdm=args.lambda_gdm,
            lambda_nonneg=args.lambda_nonneg,
            log_transform=args.log_targets
        ).to(device)
        
    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Model naming logic
    if args.model_name is None:
        parts = []
        if args.use_image and args.use_meta:
            if args.use_node:
                parts.append("multimodal_node")
            else:
                parts.append("multimodal")
        elif args.use_image:
            parts.append("image_only")
        else:
            if args.use_node:
                parts.append("meta_only_node")
            else:
                parts.append("meta_only")
            
        if args.neuro_symbolic_v3:
            parts.append("neuro_symbolic_v3")
        elif args.neuro_symbolic:
            parts.append("neurosynthetic")
        else:
            parts.append("neural")
            
        if args.log_targets:
            parts.append("log")
            
        parts.append(f"fold{args.fold}")
        args.model_name = "_".join(parts)
        
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{args.model_name}.pth")
    print(f"Model name: {args.model_name}")
    print(f"Checkpoint save path: {checkpoint_path}")
    
    # Logging history
    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Handle unfreezing backbone mid-training
        if args.unfreeze_epoch != -1 and epoch == args.unfreeze_epoch and args.use_image:
            model.unfreeze_backbone()
            # Recreate optimizer to include backbone parameters if they were frozen but now require grad
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params, lr=args.lr * 0.1, weight_decay=Config.WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
        # Train epoch
        train_loss = train_one_epoch(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer, 
            pred_criterion=pred_criterion, 
            sym_criterion=sym_criterion, 
            device=device, 
            neuro_symbolic=args.neuro_symbolic,
            neuro_symbolic_v3=args.neuro_symbolic_v3
        )
        
        # Evaluate validation
        _, _, val_metrics = evaluate_model(model, val_loader, device, log_transform=args.log_targets)
        val_loss = val_metrics['Mean_MAE']  # using Mean MAE as validation tracking target
        
        scheduler.step(val_loss)
        
        epoch_log = {
            'epoch': epoch + 1,
            'train_pred_loss': train_loss['loss_pred'],
            'train_sym_loss': train_loss['loss_sym'],
            'train_total_loss': train_loss['loss_total'],
            'val_mean_mae': val_metrics['Mean_MAE'],
            'val_mean_rmse': val_metrics['Mean_RMSE'],
            'val_add_violation': val_metrics['Mean_Add_Violation'],
            'val_nonneg_violation': val_metrics['Fraction_Negative_Predictions']
        }
        history.append(epoch_log)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss['loss_total']:.4f} (Pred: {train_loss['loss_pred']:.4f}, Sym: {train_loss['loss_sym']:.4f}) | "
              f"Val MAE: {val_metrics['Mean_MAE']:.4f} | "
              f"Add Viol: {val_metrics['Mean_Add_Violation']:.4f}")
              
        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Saved best model checkpoint!")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Final evaluation using best model
    print("\nTraining complete. Evaluating best model on validation split...")
    model.load_state_dict(torch.load(checkpoint_path))
    val_preds, val_targets, val_final_metrics = evaluate_model(model, val_loader, device, log_transform=args.log_targets)
    
    print("\n--- Final Validation Metrics ---")
    for k, v in val_final_metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save training history and final metrics
    results = {
        'config': vars(args),
        'metrics': val_final_metrics,
        'history': history
    }
    
    # Helper to recursively convert NumPy types to standard Python types for JSON
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
        
    results = convert_types(results)
    
    results_path = os.path.join(Config.CHECKPOINT_DIR, f"{args.model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results and metadata to: {results_path}")
    
    # Save predictions for post-analysis
    preds_df = pd.DataFrame(val_preds, columns=[f"pred_{t}" for t in Config.TARGETS])
    targets_df = pd.DataFrame(val_targets, columns=[f"true_{t}" for t in Config.TARGETS])
    
    # Get pivoted wide validation rows to keep metadata matched with predictions
    wide_csv = pd.read_csv(Config.TRAIN_WIDE_CSV)
    val_meta_df = wide_csv[wide_csv['fold'] == args.fold].reset_index(drop=True)
    
    eval_predictions_df = pd.concat([val_meta_df, preds_df, targets_df], axis=1)
    preds_csv_path = os.path.join(Config.CHECKPOINT_DIR, f"{args.model_name}_predictions.csv")
    eval_predictions_df.to_csv(preds_csv_path, index=False)
    print(f"Saved prediction outputs to: {preds_csv_path}")

if __name__ == '__main__':
    main()
