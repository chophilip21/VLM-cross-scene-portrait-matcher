"""Sanity check functions for the existence of model weights."""
import cv2
from loguru import logger
import numpy as np
import os

def embeddings_sanity_check(cleaned_embeddings, save_path_dir):
    """Sanity check functions for the existence of model weights."""
    for embedding_info in cleaned_embeddings:
        image_path = embedding_info['image_path']
        box = embedding_info['box']
        mask = embedding_info['mask']
        box_index = embedding_info['box_index']

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image {image_path}")
            continue

        # Ensure the mask is the same size as the image
        mask_resized = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create a color mask (e.g., red color)
        color_mask = np.zeros_like(img)
        color_mask[:, :, 2] = 255  # Red channel

        # Apply the mask to the color mask
        masked_color = cv2.bitwise_and(color_mask, color_mask, mask=mask_resized)

        # Overlay the mask on the image
        img_masked = cv2.addWeighted(img, 1.0, masked_color, 0.5, 0)

        # Draw the bounding box
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_masked, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Save the image
        image_filename = os.path.basename(image_path)
        save_filename = f"{os.path.splitext(image_filename)[0]}_box{box_index}.jpg"
        save_path = os.path.join(save_path_dir, save_filename)
        cv2.imwrite(save_path, img_masked)
        logger.info(f"Saved debug image: {save_path}")