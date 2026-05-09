import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

def crop_and_resize_faces(input_dir, output_dir, size=(224, 224)):
    detector = MTCNN()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in os.listdir(input_dir):
        class_input_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_input_dir):
            continue
            
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        print(f"Processing faces for {class_name}...")
        
        for filename in tqdm(os.listdir(class_input_dir)):
            input_path = os.path.join(class_input_dir, filename)
            output_path = os.path.join(class_output_dir, filename)
            
            if os.path.exists(output_path):
                continue
                
            try:
                img = cv2.imread(input_path)
                if img is None:
                    continue
                    
                # Convert BGR to RGB for MTCNN
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(img_rgb)
                
                if results:
                    # Pick the largest bounding box (most likely the main subject)
                    bounding_box = sorted(results, key=lambda b: b['box'][2] * b['box'][3], reverse=True)[0]['box']
                    
                    x, y, w, h = bounding_box
                    x, y = max(0, x), max(0, y)
                    
                    face = img[y:y+h, x:x+w]
                    
                    if face.size > 0:
                        face_resized = cv2.resize(face, size)
                        cv2.imwrite(output_path, face_resized)
            except Exception as e:
                pass

if __name__ == "__main__":
    cleaned_dir = os.path.join("data", "cleaned")
    cropped_dir = os.path.join("data", "cropped_faces")
    
    if not os.path.exists(cleaned_dir):
        print(f"Directory {cleaned_dir} does not exist. Run scraping and cleaning first.")
    else:
        crop_and_resize_faces(cleaned_dir, cropped_dir)
