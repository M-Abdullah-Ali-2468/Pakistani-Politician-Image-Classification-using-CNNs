import os

def print_separator():
    print("-" * 50)

if __name__ == "__main__":
    print_separator()
    print("Welcome to Pakistani Politician Image Classification Dataset Pipeline!")
    print_separator()
    print("Please run the scripts in the following order:")
    print("\n[STEP 1] Scraping Images")
    print("Run: python scripts/scraping/scrape_images.py")
    print("\n[STEP 2] Manual Cleaning")
    print("Copy images from data/raw/ to data/cleaned/ and manually remove incorrect/bad images.")
    print("\n[STEP 3] Remove Duplicates")
    print("Run: python scripts/cleaning/remove_duplicates.py")
    print("\n[STEP 4] Face Cropping (MTCNN)")
    print("Run: python scripts/face_crop/crop_faces.py")
    print("\n[STEP 5] Manual Verification")
    print("Check data/cropped_faces/ to ensure faces were cropped correctly.")
    print("\n[STEP 6] Split Dataset (Train/Val/Test)")
    print("Run: python scripts/splitting/split_dataset.py")
    print_separator()
