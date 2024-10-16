"""Sanity check functions for the existence of model weights."""

import cv2
from loguru import logger
import numpy as np
import os
from typing import List, Dict


def embeddings_sanity_check(cleaned_embeddings: List[Dict], save_path_dir: str):
    """Sanity check function that visualizes embeddings with cluster labels."""
    for embedding_info in cleaned_embeddings:
        image_path = embedding_info["image_path"]
        box = embedding_info["box"]
        mask = embedding_info["mask"]
        box_index = embedding_info["box_index"]
        cluster_label = embedding_info.get("cluster_label", -1)

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image {image_path}")
            continue

        # Resize mask to image size
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Create a red color mask
        color_mask = np.zeros_like(img)
        color_mask[:, :, 2] = 255  # Red channel

        # Apply the mask
        masked_color = cv2.bitwise_and(color_mask, color_mask, mask=mask_resized)

        # Overlay the mask
        img_masked = cv2.addWeighted(img, 1.0, masked_color, 0.5, 0)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_masked, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Add cluster label text
        cv2.putText(
            img_masked,
            f"Cluster: {cluster_label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # Save the image
        image_filename = os.path.basename(image_path)
        save_filename = f"{os.path.splitext(image_filename)[0]}_box{box_index}_cluster{cluster_label}.jpg"
        save_path = os.path.join(save_path_dir, save_filename)
        cv2.imwrite(save_path, img_masked)
        logger.info(f"Saved debug image: {save_path}")
