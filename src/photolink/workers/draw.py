"""Sanity check functions for the existence of model weights."""

import os
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from loguru import logger
from sklearn.manifold import TSNE


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




def visualize_embeddings_tsne_interactive(embeddings_info: List[Dict], save_path: str):
    """
    Visualize embeddings using t-SNE interactively and save the plot as an HTML file.

    Args:
        embeddings_info (list): List of embeddings with cluster labels.
        save_path (str): Directory to save the t-SNE plot.
    """
    # Extract embeddings, labels, and image paths
    embeddings = np.array([item["embedding"] for item in embeddings_info])
    embeddings = np.squeeze(embeddings, axis=1)
    labels = np.array([item.get("cluster_label", -1) for item in embeddings_info])
    image_paths = [item["image_path"] for item in embeddings_info]

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings)

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels,
        'image_path': image_paths
    })

    # Create interactive plot
    fig = px.scatter(
        df, x='x', y='y', color='label',
        hover_data=['image_path'],
        title='t-SNE Interactive Visualization of Embeddings'
    )

    # Save interactive plot
    plot_path = os.path.join(save_path, "tsne_plot_interactive.html")
    fig.write_html(plot_path)
    print(f"Interactive t-SNE plot saved to {plot_path}")
