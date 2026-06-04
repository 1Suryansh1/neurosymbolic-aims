import os
import glob
from PIL import Image

def main():
    src_dir = os.path.join('data', 'raw', 'train')
    dst_dir = os.path.join('data', 'processed', 'train_224')
    os.makedirs(dst_dir, exist_ok=True)
    
    img_paths = glob.glob(os.path.join(src_dir, '*.jpg'))
    total_images = len(img_paths)
    print(f"Resizing {total_images} images to 224x224 and saving to {dst_dir}...")
    
    for i, p in enumerate(img_paths):
        name = os.path.basename(p)
        dst_p = os.path.join(dst_dir, name)
        if not os.path.exists(dst_p):
            try:
                img = Image.open(p).convert('RGB')
                img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
                img_resized.save(dst_p, quality=90)
            except Exception as e:
                print(f"Error resizing {p}: {e}")
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total_images} images...")
            
    print("Pre-resizing of images complete!")

if __name__ == '__main__':
    main()
