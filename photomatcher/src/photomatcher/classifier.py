from dotenv import load_dotenv
import os
env_file = os.path.join(os.path.dirname(__file__), "resources/config.env")
load_dotenv(env_file)

import cv2
import photomatcher.model.yunet as yunet
import photomatcher.utils as utils
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
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

def expand_bounding_box(face, expansion_factor=0.3):
    """Expand the bounding box of the face upwards."""
    x, y, w, h = face[:4]
    expand_y = int(h * expansion_factor)
    y_new = max(0, y - expand_y)
    return (int(x), int(y_new), int(w), int(h + expand_y))

def extract_features(region):
    """Extract features from an image region using MobileNetV3 Small."""
    image = Image.fromarray(region).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = mobilenet_v3_small(image).numpy().flatten()
    return features

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
            face_region = image[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]

            # Visualize the expanded face region
            cv2.rectangle(image, (x_exp, y_exp), (x_exp+w_exp, y_exp+h_exp), (255, 0, 0), 2)
            cv2.imshow('Face Region', image)
            cv2.waitKey(1)  # Display the image for 1ms

            # Ask for user input to label the face
            label = input(f"Is this the main subject in {image_path}? (1 for yes, 0 for no): ")
            label = int(label)

            # Extract features and create feature vector
            features = extract_features(face_region)
            feature_vector = np.concatenate(([image_width, image_height, x, y, w, h], features))
            X.append(feature_vector)
            Y.append(label)

            # Close the displayed image
            cv2.destroyAllWindows()

         # Save progress to cache file
        np.savez(cache_file, X=np.array(X), y=np.array(Y))
        print(f"Saved {cache_file}")

    return True

if __name__ == "__main__":
    dataset_path_a = '/home/chophilip21/togamatcher/dataset/Secondary/On Stage'
    output_path = '/home/chophilip21/togamatcher/dataset/training_data'

    create_dataset(dataset_path_a, output_path)
