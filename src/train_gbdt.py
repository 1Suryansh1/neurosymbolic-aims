import os
import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from src.config import Config

def prepare_gbdt_data(df_train, df_val, log_targets=False):
    # Parse dates
    for df in [df_train, df_val]:
        date = pd.to_datetime(df['Sampling_Date'])
        df['Month'] = date.dt.month
        df['DayOfYear'] = date.dt.dayofyear
        
    numeric_cols = Config.NUMERICAL_COLS + ['Month', 'DayOfYear']
    categorical_cols = Config.CATEGORICAL_COLS
    
    # Scale numericals
    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    
    df_train_scaled[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_val_scaled[numeric_cols] = scaler.transform(df_val[numeric_cols])
    
    # One-hot encode categoricals
    combined_cats = pd.concat([df_train[categorical_cols], df_val[categorical_cols]], axis=0)
    combined_encoded = pd.get_dummies(combined_cats, columns=categorical_cols, dtype=float)
    
    train_encoded = combined_encoded.iloc[:len(df_train)].reset_index(drop=True)
    val_encoded = combined_encoded.iloc[len(df_train):].reset_index(drop=True)
    
    # Merge scaled and encoded columns
    X_train = pd.concat([df_train_scaled[numeric_cols].reset_index(drop=True), train_encoded], axis=1)
    X_val = pd.concat([df_val_scaled[numeric_cols].reset_index(drop=True), val_encoded], axis=1)
    
    y_train = df_train[Config.TARGETS].values.astype(np.float32)
    y_val = df_val[Config.TARGETS].values.astype(np.float32)
    
    if log_targets:
        y_train = np.log1p(np.maximum(0.0, y_train))
        
    return X_train, y_train, X_val, y_val

def main():
    parser = argparse.ArgumentParser(description="Train GBDT (XGBoost) Tabular Baseline")
    parser.add_argument('--fold', type=int, default=0, help='CV Fold to train')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--log_targets', action='store_true', default=False, help='Log1p transform targets')
    args = parser.parse_args()
    
    csv_path = Config.TRAIN_WIDE_CSV
    if not os.path.exists(csv_path):
        print(f"Error: Wide CSV not found at {csv_path}. Run preprocessing first.")
        return
        
    df_all = pd.read_csv(csv_path)
    df_train = df_all[df_all['fold'] != args.fold].copy()
    df_val = df_all[df_all['fold'] == args.fold].copy()
    
    X_train, y_train, X_val, y_val = prepare_gbdt_data(df_train, df_val, args.log_targets)
    
    print(f"Training XGBoost baseline on fold {args.fold} (log_targets={args.log_targets})...")
    print(f"Train features shape: {X_train.shape}, Val features shape: {X_val.shape}")
    
    # Train 5 separate regressor models (one for each target)
    models = []
    val_preds = np.zeros_like(y_val)
    
    for i, target_name in enumerate(Config.TARGETS):
        print(f"Training XGBoost for target: {target_name}")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train[:, i])
        models.append(model)
        val_preds[:, i] = model.predict(X_val)
        
    if args.log_targets:
        val_preds_raw = np.expm1(np.clip(val_preds, -1.0, 10.0))
    else:
        val_preds_raw = val_preds
        
    # Evaluate
    metrics = {}
    mae_list = []
    rmse_list = []
    
    for i, name in enumerate(Config.TARGETS):
        mae = mean_absolute_error(y_val[:, i], val_preds_raw[:, i])
        rmse = root_mean_squared_error(y_val[:, i], val_preds_raw[:, i])
        r2 = r2_score(y_val[:, i], val_preds_raw[:, i])
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        
        metrics[f"{name}_MAE"] = mae
        metrics[f"{name}_RMSE"] = rmse
        metrics[f"{name}_R2"] = r2
        
    metrics['Mean_MAE'] = np.mean(mae_list)
    metrics['Mean_RMSE'] = np.mean(rmse_list)
    
    # Rule consistency metrics
    clover_pred = val_preds[:, 0]
    dead_pred = val_preds[:, 1]
    green_pred = val_preds[:, 2]
    total_pred = val_preds[:, 3]
    gdm_pred = val_preds[:, 4]
    
    # Additive violation
    calc_total_pred = clover_pred + dead_pred + green_pred
    add_violation = np.abs(total_pred - calc_total_pred)
    metrics['Mean_Add_Violation'] = np.mean(add_violation)
    metrics['Max_Add_Violation'] = np.max(add_violation)
    metrics['Add_Violation_lt_0_1'] = np.mean(add_violation < 0.1)
    metrics['Add_Violation_lt_1_0'] = np.mean(add_violation < 1.0)
    metrics['Add_Violation_lt_5_0'] = np.mean(add_violation < 5.0)
    
    # GDM-Green discrepancy
    gdm_green_disc = np.abs(gdm_pred - green_pred)
    metrics['Mean_GDM_Green_Discrepancy'] = np.mean(gdm_green_disc)
    
    # Negative predictions fraction
    metrics['Fraction_Negative_Predictions'] = np.mean(val_preds < 0)
    
    print("\n--- XGBoost Validation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save results
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    results = {
        'config': {'model': 'xgboost', 'fold': args.fold},
        'metrics': metrics
    }
    
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
    
    if args.model_name is None:
        model_name = f"xgb_meta_only_fold{args.fold}"
    else:
        model_name = args.model_name
        
    results_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Save predictions
    preds_df = pd.DataFrame(val_preds_raw, columns=[f"pred_{t}" for t in Config.TARGETS])
    targets_df = pd.DataFrame(y_val, columns=[f"true_{t}" for t in Config.TARGETS])
    val_meta_df = df_val.reset_index(drop=True)
    
    eval_predictions_df = pd.concat([val_meta_df, preds_df, targets_df], axis=1)
    preds_csv_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_predictions.csv")
    eval_predictions_df.to_csv(preds_csv_path, index=False)
    print(f"Saved GBDT predictions to: {preds_csv_path}")

if __name__ == '__main__':
    main()
