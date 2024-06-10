"""Annotate the data with the main subject in the image."""
import os
from dotenv import load_dotenv

env_file = os.path.join(os.path.dirname(__file__), "resources/config.env")
load_dotenv(env_file)

import cv2
import photomatcher.model.yunet as yunet
import photomatcher.utils as utils
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import joblib
import copy
import warnings
warnings.filterwarnings("ignore")

# Load pre-trained MobileNetV3 Small model
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
mobilenet_v3_small = mobilenet_v3_small.eval() 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def expand_bounding_box(face, expansion_factor=0.4):
    """Expand the bounding box of the face in all directions."""
    x, y, w, h = face[:4]
    expand_w = int(w * expansion_factor)
    expand_h = int(h * expansion_factor)
    x_new = max(0, x - expand_w)
    y_new = max(0, y - expand_h)
    w_new = w + 2 * expand_w
    h_new = h + 2 * expand_h
    return (int(x_new), int(y_new), int(w_new), int(h_new))

def extract_features(region):
    """Extract features from an image region using MobileNetV3 Small."""
    image = Image.fromarray(region).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = mobilenet_v3_small(image).numpy().flatten()
    return features

def display_image_with_bbox(image, bbox):
    """Display an image with bounding box using PIL."""
    x, y, w, h = bbox
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    img_pil.show()

def create_dataset(image_path: str, output_path: str):
    """Create dataset from the given path iteratively."""
    
    X = []
    Y = []

    images = utils.search_all_images(image_path)
    os.makedirs(output_path, exist_ok=True)

    for image_path in images:
        cache_file = os.path.join(output_path, f'{os.path.basename(image_path)}.npz')

        if os.path.exists(cache_file):
            print(f"Skipping {image_path}, already processed.")
            continue

        face_table = yunet.run_face_detection(image_path)
        faces = face_table['faces']
        original_image = face_table['image']
        image_height, image_width, _ = original_image.shape
       
        faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[:3]

        print(f"Processing {image_path} with {len(faces)} faces.")

        for face in faces:
            x, y, w, h = face[:4]

            image = copy.deepcopy(original_image)
            expanded_face = expand_bounding_box(face)
            x_exp, y_exp, w_exp, h_exp = expanded_face

            # Ensure expanded face region is within image boundaries
            x_exp = max(0, min(x_exp, image_width - w_exp))
            y_exp = max(0, min(y_exp, image_height - h_exp))

            face_region = image[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]

            # Visualize the expanded face region
            display_image_with_bbox(image, (x_exp, y_exp, w_exp, h_exp))

            # Ask for user input to label the face
            label = input(f"Is this the main subject in {image_path}? (1 for yes, 0 for no): ")
            label = int(label)

            # Extract features and create feature vector
            features = extract_features(face_region)
            feature_vector = np.concatenate(([image_width, image_height, x, y, w, h], features))
            print(f"Image width: {image_width}, Image height: {image_height}")
            print(f"Face bounding box: {x}, {y}, {w}, {h}")
            print(f"Extracted features: {features.shape}")
            print(f"Concatenated feature vector: {feature_vector.shape}")
            print('-' * 50)


            X.append(feature_vector)
            Y.append(label)

        # Save progress to cache file
        np.savez(cache_file, X=np.array(X), y=np.array(Y))
        print(f"Saved {cache_file}")

    return True

if __name__ == "__main__":
    dataset_path_a = '/home/chophilip21/togamatcher/dataset/Secondary/On Stage'
    output_path = '/home/chophilip21/togamatcher/dataset/training_data'

    create_dataset(dataset_path_a, output_path)
