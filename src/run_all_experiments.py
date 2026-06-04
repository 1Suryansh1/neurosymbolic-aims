import os
import subprocess
import json
import pandas as pd

def run_cmd(cmd):
    print(f"\n==================================================")
    print(f"RUNNING: {' '.join(cmd)}")
    print(f"==================================================")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
    return result.returncode

def main():
    experiments = [
        # 1. GBDT Baseline
        ["python", "-u", "-m", "src.train_gbdt", "--fold", "0"],
        
        # 2. MLP Metadata-only
        ["python", "-u", "-m", "src.train", "--use_meta", "--epochs", "50", "--fold", "0", "--model_name", "meta_only_neural_fold0"],
        
        # 3. Image-only CNN
        ["python", "-u", "-m", "src.train", "--use_image", "--epochs", "30", "--fold", "0", "--model_name", "image_only_neural_fold0"],
        
        # 4. Multimodal Neural-only Fusion
        ["python", "-u", "-m", "src.train", "--use_image", "--use_meta", "--epochs", "30", "--fold", "0", "--model_name", "multimodal_neural_fold0"],
        
        # 5. Multimodal Neuro-symbolic V1 (Additivity + Non-negativity, lambda_gdm = 0)
        ["python", "-u", "-m", "src.train", "--use_image", "--use_meta", "--neuro_symbolic", "--lambda_gdm", "0", "--epochs", "30", "--fold", "0", "--model_name", "neuro_symbolic_v1_fold0"]
    ]
    
    # Run all sequentially
    for cmd in experiments:
        run_cmd(cmd)
        
    print("\n==================================================")
    print("ALL EXPERIMENTS COMPLETED. COMPILING RESULTS...")
    print("==================================================")
    
    # Compile table
    model_files = {
        "GBDT (XGBoost) Tabular Baseline": "xgb_meta_only_fold0_results.json",
        "MLP Metadata-only Baseline": "meta_only_neural_fold0_results.json",
        "Image-only CNN Baseline": "image_only_neural_fold0_results.json",
        "Multimodal Neural-only Fusion": "multimodal_neural_fold0_results.json",
        "Multimodal Neuro-symbolic V1 (Add + Non-neg)": "neuro_symbolic_v1_fold0_results.json"
    }
    
    checkpoint_dir = 'checkpoints'
    summary_data = []
    
    for model_name, filename in model_files.items():
        path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    res = json.load(f)
                metrics = res['metrics']
                summary_data.append({
                    "Model": model_name,
                    "Mean MAE (g)": f"{metrics['Mean_MAE']:.4f}",
                    "Mean RMSE (g)": f"{metrics['Mean_RMSE']:.4f}",
                    "Mean Additive Violation (g)": f"{metrics['Mean_Add_Violation']:.4f}",
                    "GDM-Green Discrepancy (g)": f"{metrics['Mean_GDM_Green_Discrepancy']:.4f}",
                    "Fraction Neg Predictions": f"{metrics['Fraction_Negative_Predictions']:.4f}"
                })
            except Exception as e:
                print(f"Error reading {path}: {e}")
        else:
            summary_data.append({
                "Model": model_name,
                "Mean MAE (g)": "N/A",
                "Mean RMSE (g)": "N/A",
                "Mean Additive Violation (g)": "N/A",
                "GDM-Green Discrepancy (g)": "N/A",
                "Fraction Neg Predictions": "N/A"
            })
            
    df_summary = pd.DataFrame(summary_data)
    print("\n--- Summary Performance Table ---")
    print(df_summary.to_string(index=False))
    
    # Save markdown summary report
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    summary_md_path = os.path.join(reports_dir, 'ablation_summary.md')
    with open(summary_md_path, 'w') as f:
        f.write("# Ablation Summary Report\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n")
    print(f"\nSaved summary markdown to: {summary_md_path}")

if __name__ == '__main__':
    main()
