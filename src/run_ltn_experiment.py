import subprocess
import os

print("Running 5-fold CV for Multimodal Neuro-symbolic V3 (LTN) in GRAM SPACE...")
for fold in range(5):
    print(f"\n--- Starting Fold {fold} ---")
    
    cmd = [
        "python", "-m", "src.train",
        "--fold", str(fold),
        "--neuro_symbolic_v3",
        # NO --log_targets! Proving scale invariance of LTN
        "--epochs", "30",
        "--lambda_add", "1.0", # LTN truth is [0, 1]
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"--- Finished Fold {fold} ---")
    except subprocess.CalledProcessError as e:
        print(f"Error during training fold {fold}: {e}")
        break

print("\nFinished 5-fold CV for LTN model.")
