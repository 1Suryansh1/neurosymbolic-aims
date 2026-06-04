import os
import pandas as pd
import numpy as np

def main():
    # Define paths
    raw_csv_path = 'train (1).csv'
    raw_img_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Reading raw CSV from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    
    # Check image existence
    # In the CSV, image_path is 'train/IDxxxxxxxx.jpg'
    # The zip is extracted as data/raw/train/IDxxxxxxxx.jpg
    df['full_img_path'] = df['image_path'].apply(lambda x: os.path.join(raw_img_dir, x.replace('/', os.sep)))
    df['image_exists'] = df['full_img_path'].apply(os.path.exists)
    
    total_rows = len(df)
    existing_rows = df['image_exists'].sum()
    print(f"Total rows in CSV: {total_rows}")
    print(f"Rows with existing images: {existing_rows} ({existing_rows / total_rows * 100:.2f}%)")
    
    # Filter for rows where image exists
    df_filtered = df[df['image_exists']].copy()
    
    # Let's check unique image paths
    unique_paths_all = df['image_path'].nunique()
    unique_paths_filtered = df_filtered['image_path'].nunique()
    print(f"Unique images in CSV: {unique_paths_all}")
    print(f"Unique images available: {unique_paths_filtered}")
    
    # Pivot target values into wide format
    index_cols = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    df_wide = df_filtered.pivot(index=index_cols, columns='target_name', values='target').reset_index()
    
    # Target columns
    targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Check if we have all targets for each pivoted row
    # (Since there are 5 targets per image, df_wide should have no NaNs in target columns)
    missing_targets = df_wide[targets].isna().sum()
    print("Missing values in targets after pivoting:")
    print(missing_targets)
    
    # Verify additivity constraint: Dry_Total_g ≈ Dry_Clover_g + Dry_Dead_g + Dry_Green_g
    calc_total = df_wide['Dry_Clover_g'] + df_wide['Dry_Dead_g'] + df_wide['Dry_Green_g']
    diff = (df_wide['Dry_Total_g'] - calc_total).abs()
    print(f"Additive check - Mean absolute difference: {diff.mean():.6f} g")
    print(f"Additive check - Max absolute difference: {diff.max():.6f} g")
    print(f"Percentage of samples with difference < 0.01 g: {(diff < 0.01).mean() * 100:.2f}%")
    
    # Save processed wide dataset
    output_path = os.path.join(processed_dir, 'train_wide.csv')
    df_wide.to_csv(output_path, index=False)
    print(f"Saved wide pivoted data to {output_path} (Shape: {df_wide.shape})")

if __name__ == '__main__':
    main()
