import os
import imagehash
from PIL import Image
from tqdm import tqdm

def remove_duplicates(input_dir):
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Cleaning duplicates in {class_name}...")
        hashes = set()
        removed_count = 0
        
        for filename in tqdm(os.listdir(class_dir)):
            filepath = os.path.join(class_dir, filename)
            try:
                with Image.open(filepath) as img:
                    # Calculate perceptual hash
                    img_hash = imagehash.phash(img)
                    
                if img_hash in hashes:
                    # Duplicate found
                    os.remove(filepath)
                    removed_count += 1
                else:
                    hashes.add(img_hash)
            except Exception as e:
                # Remove corrupted images
                try:
                    os.remove(filepath)
                    removed_count += 1
                except:
                    pass
                    
        print(f"Removed {removed_count} duplicate/corrupted images from {class_name}.\n")

if __name__ == "__main__":
    cleaned_dir = os.path.join("data", "cleaned")
    
    if not os.path.exists(cleaned_dir):
        print(f"Directory {cleaned_dir} does not exist.")
        print("Please copy your manually cleaned images into this directory first.")
    else:
        remove_duplicates(cleaned_dir)
