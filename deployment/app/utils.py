import torch
from PIL import Image, ImageOps
from torchvision import transforms
from io import BytesIO

# Standards for most AI models
IMAGE_SIZE = 224
CLASS_NAMES = [
    "ahmed_sharif_chaudhry", "asif_ali_zardari", "benazir_bhutto",
    "bilawal_bhutto", "fazl_ur_rehman", "hina_rabbani_khar",
    "imran_khan", "khawaja_asif", "khawaja_saad_rafique",
    "maryam_nawaz", "nawaz_sharif", "pervez_musharraf",
    "shah_mahmood_qureshi", "shehbaz_sharif", "sheikh_rasheed",
    "yousuf_raza_gillani",
]

def get_transform():
    """Defines the steps to resize and clean the image."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def prepare_image(image_bytes: bytes) -> torch.Tensor:
    """Converts raw upload bytes into a math tensor."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image) # Fixes image rotation issues
    transform = get_transform()
    return transform(image).unsqueeze(0) # Add a 'batch' dimension