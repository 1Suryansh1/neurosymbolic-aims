import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    REPORT_DIR = os.path.join(BASE_DIR, 'reports')
    
    # Files
    TRAIN_WIDE_CSV = os.path.join(PROCESSED_DATA_DIR, 'train_wide_splits.csv')
    
    # Dataset Parameters
    IMAGE_SIZE = (224, 224)
    TARGETS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    NUM_TARGETS = len(TARGETS)
    
    # Preprocessing
    CATEGORICAL_COLS = ['Species', 'State']
    NUMERICAL_COLS = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    # Model Hyperparameters
    BACKBONE = 'resnet18'  # or 'resnet34', 'efficientnet_b0'
    FREEZE_BACKBONE = True
    IMAGE_EMB_DIM = 256
    META_EMB_DIM = 64
    FUSION_DIM = 128
    
    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    BACKBONE_LR = 1e-4  # learning rate for backbone if unfrozen
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    # Loss weights (Neuro-symbolic options)
    PRED_LOSS_TYPE = 'huber'  # 'mse' or 'huber'
    
    # Rule Weights
    LAMBDA_ADD = 1.0       # Dry_Total_g ≈ Clover + Dead + Green
    LAMBDA_GDM = 0.2       # GDM_g ≈ Dry_Green_g
    LAMBDA_NONNEG = 0.1    # Biomass >= 0
    
    # Option for log1p transformation of targets
    LOG_TRANSFORM_TARGETS = False  # Try both True and False in ablations

# Ensure directories exist
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
