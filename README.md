# Pakistani Politician Image Classification using CNNs

This project aims to classify images of 16 Pakistani politicians using Convolutional Neural Networks.

## Dataset Collection Pipeline

The data collection phase is split into multiple scripts to ensure high quality and accuracy.

### 1. Environment Setup

```bash
# Create Virtual Environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

The pipeline is split into multiple automated and manual steps:

1. **Scraping**: Run `python scripts/scraping/scrape_images.py` to download raw images from Google.
2. **Manual Cleaning**: Copy images from `data/raw/` to `data/cleaned/` and manually delete incorrect, blurry, or group photos.
3. **Remove Duplicates**: Run `python scripts/cleaning/remove_duplicates.py` to remove duplicate and exact similar images using image hashing.
4. **Face Crop**: Run `python scripts/face_crop/crop_faces.py` to extract and resize faces to 224x224 using MTCNN.
5. **Split Dataset**: Run `python scripts/splitting/split_dataset.py` to create train (75%), val (15%), and test (10%) sets.

You can also run `python main.py` for a quick guide on the steps.