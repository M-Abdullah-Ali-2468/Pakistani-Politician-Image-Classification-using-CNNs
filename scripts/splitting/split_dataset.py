import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.75, val_ratio=0.15):
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        images = os.listdir(class_dir)
        random.shuffle(images)
        
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        for split, split_imgs in zip(splits, [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img in split_imgs:
                src = os.path.join(class_dir, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy(src, dst)
                
        print(f"Splitted {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test.")

if __name__ == "__main__":
    cropped_dir = os.path.join("data", "cropped_faces")
    final_dir = os.path.join("data", "final_dataset")
    
    if not os.path.exists(cropped_dir):
        print(f"Directory {cropped_dir} does not exist. Run face cropping first.")
    else:
        random.seed(42)
        split_dataset(cropped_dir, final_dir)
