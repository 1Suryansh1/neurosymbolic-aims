import os
import subprocess
import json
import numpy as np
import pandas as pd
from src.config import Config

def run_cmd(cmd):
    print(f"\nRUNNING: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}: {result.stderr}")
    return result.returncode

def main():
    folds = [0, 1, 2, 3, 4]
    
    # Define the 6 models and their train configurations
    # We will run them across all 5 folds
    model_configs = [
        # 1. GBDT Baseline
        {"name": "GBDT (XGBoost) Tabular Baseline", "prefix": "xgb_meta_only", "cmd_base": ["python", "-u", "-m", "src.train_gbdt"]},
        
        # 2. MLP Metadata-only
        {"name": "MLP Metadata-only Baseline", "prefix": "meta_only_neural", "cmd_base": ["python", "-u", "-m", "src.train", "--use_meta", "--epochs", "50"]},
        
        # 3. Image-only CNN
        {"name": "Image-only CNN Baseline", "prefix": "image_only_neural", "cmd_base": ["python", "-u", "-m", "src.train", "--use_image", "--epochs", "30"]},
        
        # 4. Multimodal Neural-only Fusion
        {"name": "Multimodal Neural-only Fusion", "prefix": "multimodal_neural", "cmd_base": ["python", "-u", "-m", "src.train", "--use_image", "--use_meta", "--epochs", "30"]},
        
        # 5. Multimodal Neuro-symbolic V1 (Add + Non-neg)
        {"name": "Multimodal Neuro-symbolic V1 (Add + Non-neg)", "prefix": "neuro_symbolic_v1", "cmd_base": ["python", "-u", "-m", "src.train", "--use_image", "--use_meta", "--neuro_symbolic", "--lambda_gdm", "0", "--epochs", "30"]}
    ]
    
    # Run all folds for all models
    total_runs = len(model_configs) * len(folds)
    current_run = 0
    
    for m_cfg in model_configs:
        for fold in folds:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Training {m_cfg['name']} on Fold {fold}...")
            
            # Setup specific run
            model_name = f"{m_cfg['prefix']}_fold{fold}"
            cmd = m_cfg['cmd_base'] + ["--fold", str(fold), "--model_name", model_name]
            run_cmd(cmd)
            
    print("\n==================================================")
    print("ALL FOLDS COMPLETED. COMPILING CROSS-VALIDATION RESULTS...")
    print("==================================================")
    
    checkpoint_dir = 'checkpoints'
    summary_data = []
    
    for m_cfg in model_configs:
        fold_maes = []
        fold_rmses = []
        fold_add_viols = []
        fold_gdm_greens = []
        fold_negs = []
        
        for fold in folds:
            filename = f"{m_cfg['prefix']}_fold{fold}_results.json"
            path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        res = json.load(f)
                    metrics = res['metrics']
                    fold_maes.append(metrics['Mean_MAE'])
                    fold_rmses.append(metrics['Mean_RMSE'])
                    fold_add_viols.append(metrics['Mean_Add_Violation'])
                    fold_gdm_greens.append(metrics['Mean_GDM_Green_Discrepancy'])
                    fold_negs.append(metrics['Fraction_Negative_Predictions'])
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    
        if len(fold_maes) == len(folds):
            summary_data.append({
                "Model": m_cfg['name'],
                "Mean MAE (g)": f"{np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}",
                "Mean RMSE (g)": f"{np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}",
                "Mean Additive Violation (g)": f"{np.mean(fold_add_viols):.4f} ± {np.std(fold_add_viols):.4f}",
                "GDM-Green Discrepancy (g)": f"{np.mean(fold_gdm_greens):.4f} ± {np.std(fold_gdm_greens):.4f}",
                "Fraction Neg Predictions": f"{np.mean(fold_negs):.4f} ± {np.std(fold_negs):.4f}"
            })
        else:
            summary_data.append({
                "Model": m_cfg['name'],
                "Mean MAE (g)": "Incomplete",
                "Mean RMSE (g)": "Incomplete",
                "Mean Additive Violation (g)": "Incomplete",
                "GDM-Green Discrepancy (g)": "Incomplete",
                "Fraction Neg Predictions": "Incomplete"
            })
            
    df_summary = pd.DataFrame(summary_data)
    print("\n--- 5-Fold Cross-Validation Performance Table ---")
    print(df_summary.to_string(index=False))
    
    # Save a JSON containing cross-validation summary
    summary_json_path = os.path.join(checkpoint_dir, 'cv_summary.json')
    df_summary.to_json(summary_json_path, orient='records', indent=4)
    
if __name__ == '__main__':
    main()
