import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision.models as models
from src.config import Config
from src.transforms import get_val_transforms

def main():
    csv_path = Config.TRAIN_WIDE_CSV
    if not os.path.exists(csv_path):
        print(f"Error: Wide CSV not found at {csv_path}. Run preprocessing first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Initialize resnet18 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extracting features using device: {device}")
    
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    except ImportError:
        backbone = models.resnet18(pretrained=True)
        
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()
    
    transform = get_val_transforms(Config.IMAGE_SIZE)
    img_dir = Config.RAW_DATA_DIR
    
    features_dict = {}
    total_images = len(df)
    print(f"Extracting image features for {total_images} images...")
    
    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            filename = os.path.basename(row['image_path'])
            resized_path = os.path.join('data', 'processed', 'train_224', filename)
            if os.path.exists(resized_path):
                img_path = resized_path
            else:
                img_path = os.path.join(img_dir, row['image_path'].replace('/', os.sep))
                
            image = Image.open(img_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            feat = backbone(img_tensor).squeeze(0).cpu()
            features_dict[row['image_path']] = feat
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{total_images} image features...")
                
    # Save the features dict
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(Config.PROCESSED_DATA_DIR, 'image_features.pt')
    torch.save(features_dict, out_path)
    print(f"Saved pre-extracted features to {out_path}")

if __name__ == '__main__':
    main()
