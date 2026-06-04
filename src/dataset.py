import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import StandardScaler
from src.config import Config
from src.transforms import get_train_transforms, get_val_transforms

class BiomassDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, targets=None, log_transform=False, load_images=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.targets = targets or Config.TARGETS
        self.log_transform = log_transform
        self.load_images = load_images
        
        # Load pre-extracted features if they exist
        features_path = os.path.join(Config.PROCESSED_DATA_DIR, 'image_features.pt')
        if os.path.exists(features_path):
            self.features_dict = torch.load(features_path)
        else:
            self.features_dict = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image if requested, else return dummy
        if self.load_images:
            if self.features_dict is not None and row['image_path'] in self.features_dict:
                image = self.features_dict[row['image_path']]
            else:
                filename = os.path.basename(row['image_path'])
                resized_path = os.path.join('data', 'processed', 'train_224', filename)
                if os.path.exists(resized_path):
                    img_path = resized_path
                else:
                    img_path = os.path.join(self.img_dir, row['image_path'].replace('/', os.sep))
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
        else:
            image = torch.zeros(3, 224, 224, dtype=torch.float32)
            
        # Get metadata vector
        meta_cols = [c for c in self.df.columns if c.startswith('meta_')]
        meta_values = self.df.loc[idx, meta_cols].values.astype(np.float32)
        meta_vector = torch.tensor(meta_values, dtype=torch.float32)
        
        # Get targets
        target_values = row[self.targets].values.astype(np.float32)
        if self.log_transform:
            target_values = np.log1p(np.maximum(0, target_values))
            
        targets_tensor = torch.tensor(target_values, dtype=torch.float32)
        
        return image, meta_vector, targets_tensor

def prepare_metadata(df_train, df_val):
    """
    Fits preprocessing objects on train split and transforms both train and val.
    """
    # 1. Parse date features
    for df in [df_train, df_val]:
        date = pd.to_datetime(df['Sampling_Date'])
        df['Month'] = date.dt.month
        df['DayOfYear'] = date.dt.dayofyear
        
    numeric_cols = Config.NUMERICAL_COLS + ['Month', 'DayOfYear']
    categorical_cols = Config.CATEGORICAL_COLS
    
    # 2. Scale numeric features
    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    
    df_train_scaled[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_val_scaled[numeric_cols] = scaler.transform(df_val[numeric_cols])
    
    # 3. One-hot encode categoricals (with alignment)
    # Combine species and state mapping to ensure all columns are present
    combined_cats = pd.concat([df_train[categorical_cols], df_val[categorical_cols]], axis=0)
    combined_encoded = pd.get_dummies(combined_cats, columns=categorical_cols, dtype=float)
    
    train_encoded = combined_encoded.iloc[:len(df_train)]
    val_encoded = combined_encoded.iloc[len(df_train):]
    
    # Rename columns to meta_ prefix for easy filtering and reset index to avoid alignment issues
    train_meta_numeric = df_train_scaled[numeric_cols].rename(columns=lambda x: f"meta_{x}").reset_index(drop=True)
    val_meta_numeric = df_val_scaled[numeric_cols].rename(columns=lambda x: f"meta_{x}").reset_index(drop=True)
    
    train_meta_cat = train_encoded.rename(columns=lambda x: f"meta_{x}").reset_index(drop=True)
    val_meta_cat = val_encoded.rename(columns=lambda x: f"meta_{x}").reset_index(drop=True)
    
    # Merge back
    df_train_final = pd.concat([df_train.reset_index(drop=True), train_meta_numeric, train_meta_cat], axis=1)
    df_val_final = pd.concat([df_val.reset_index(drop=True), val_meta_numeric, val_meta_cat], axis=1)
    
    return df_train_final, df_val_final

def get_dataloaders(fold, batch_size=None, image_size=None, log_transform_targets=None, use_image=True):
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if image_size is None:
        image_size = Config.IMAGE_SIZE
    if log_transform_targets is None:
        log_transform_targets = Config.LOG_TRANSFORM_TARGETS
        
    csv_path = Config.TRAIN_WIDE_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Wide CSV not found at {csv_path}. Run preprocessing first.")
        
    df_all = pd.read_csv(csv_path)
    
    df_train_raw = df_all[df_all['fold'] != fold].copy()
    df_val_raw = df_all[df_all['fold'] == fold].copy()
    
    # Preprocess metadata (scale and encode)
    df_train_processed, df_val_processed = prepare_metadata(df_train_raw, df_val_raw)
    
    # Get image directory
    img_dir = Config.RAW_DATA_DIR
    
    # Create datasets
    train_ds = BiomassDataset(
        df=df_train_processed,
        img_dir=img_dir,
        transform=get_train_transforms(image_size),
        log_transform=log_transform_targets,
        load_images=use_image
    )
    
    val_ds = BiomassDataset(
        df=df_val_processed,
        img_dir=img_dir,
        transform=get_val_transforms(image_size),
        log_transform=log_transform_targets,
        load_images=use_image
    )
    
    # Create dataloaders (optimize workers and pinning for CUDA vs CPU)
    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    # Metadata vector dimension
    meta_cols = [c for c in df_train_processed.columns if c.startswith('meta_')]
    meta_dim = len(meta_cols)
    
    return train_loader, val_loader, meta_dim

if __name__ == '__main__':
    # Test dataloader creation
    try:
        train_l, val_l, meta_d = get_dataloaders(fold=0, batch_size=8)
        print(f"Dataset preparation test: SUCCESS")
        print(f"Meta dimension: {meta_d}")
        img, meta, target = next(iter(train_l))
        print(f"Batch shapes - Image: {img.shape}, Meta: {meta.shape}, Targets: {target.shape}")
    except Exception as e:
        print(f"Dataset preparation test: FAILED - {e}")
