import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def main():
    processed_csv = os.path.join('data', 'processed', 'train_wide.csv')
    if not os.path.exists(processed_csv):
        print(f"Error: {processed_csv} does not exist. Run preprocess.py first.")
        return
        
    df = pd.read_csv(processed_csv)
    
    print("State counts:")
    print(df['State'].value_counts())
    print("\nSpecies counts:")
    print(df['Species'].value_counts())
    
    # Create combined stratification key
    # Some species-state pairs might be very rare, so we group them
    df['stratify_key'] = df['State'].astype(str) + "_" + df['Species'].astype(str)
    
    # Check frequency of each key
    counts = df['stratify_key'].value_counts()
    print("\nRaw stratification key counts:")
    print(counts)
    
    # Any combination with fewer than 5 samples is merged to prevent issues in 5-fold split
    rare_keys = counts[counts < 5].index
    df['stratify_key'] = df['stratify_key'].apply(lambda x: "Rare_Group" if x in rare_keys else x)
    
    print("\nStratification key counts after grouping rare classes:")
    print(df['stratify_key'].value_counts())
    
    # Perform stratified K-fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df['fold'] = -1
    
    # We stratify based on the stratify_key
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify_key'])):
        df.loc[val_idx, 'fold'] = fold
        
    # Print state/species distribution per fold to verify stratification
    for fold in range(5):
        fold_df = df[df['fold'] == fold]
        print(f"\n--- Fold {fold} (Size: {len(fold_df)}) ---")
        print("States ratio:")
        print(fold_df['State'].value_counts(normalize=True).round(3))
        
    # Save the splits to splits folder and update processed folder
    os.makedirs(os.path.join('data', 'splits'), exist_ok=True)
    
    # Save split indices for clean reference
    for fold in range(5):
        fold_train = df[df['fold'] != fold].index.tolist()
        fold_val = df[df['fold'] == fold].index.tolist()
        pd.Series(fold_train).to_csv(os.path.join('data', 'splits', f'train_fold_{fold}.txt'), index=False, header=False)
        pd.Series(fold_val).to_csv(os.path.join('data', 'splits', f'val_fold_{fold}.txt'), index=False, header=False)
        
    # Save the dataset containing fold labels
    output_path = os.path.join('data', 'processed', 'train_wide_splits.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated splits and saved wide data with folds to: {output_path}")

if __name__ == '__main__':
    main()
