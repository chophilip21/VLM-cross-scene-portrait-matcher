"""Add lower level dp2 functions to avoid cluttering the main functional modules."""

import os
from pathlib import Path

from loguru import logger

from photolink import get_application_path, get_config
from photolink.models.yolo_seg import YOLOSeg
from photolink.utils.function import safe_load_image, search_all_images


def run_yolo_seg(image_list: list) -> dict:
    """Use yoloworld to detect humans based on precomputed embeddings.

    Iterate over list of images, and return dict based on the image name as key.
    """

    if not isinstance(image_list, list):
        logger.error(f"Invalid image list: {image_list}")
        raise ValueError("Invalid image list.")

    if len(image_list) == 0:
        logger.error(f"Empty image list: {image_list}")
        raise ValueError("Empty image list.")

    application_path = get_application_path()
    config = get_config()
    model_path = str(application_path / Path(config.get("YOLOSEG", "LOCAL_PATH")))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found on config file: {model_path}")

    conf = float(config.get("YOLOSEG", "CONF"))
    iou = float(config.get("YOLOSEG", "IOU"))
    top_n = int(config.get("YOLOSEG", "TOP_N"))
    heuristics = config.get("YOLOSEG", "HEURISTIC")

    model = YOLOSeg(
        model_path,
        human_only=True,
        num_candidates=top_n,
        conf_thres=conf,
        iou_thres=iou,
        heuristic_threshold=heuristics,
    )

    output_dict = {}

    # iterate over images.
    for image_path in image_list:

        try:
            img = safe_load_image(image_path)
        except Exception as e:
            logger.error(f"Error loading image: {image_path}, {e}")
            error = f"Error loading image: {image_path}, {e}"
            output_dict["error"] = error
            continue

        try:
            boxes, segments, _ = model(img, conf_threshold=conf, iou_threshold=iou)
            output_dict[image_path] = {"boxes": boxes, "segments": segments}
        except Exception as e:
            logger.error(f"Error processing image: {image_path}, {e}")
            error = f"Error during segmentation inference on image: {image_path}, {e}"
            output_dict["error"] = error
            continue

    return output_dict


if __name__ == "__main__":

    demo_path = str(Path(r"C:\Users\choph\philip\FOR PHIL\USFU\C1\C1P1\EDITS"))
    img_list = search_all_images(demo_path)
    test = run_yolo_seg(image_list=img_list, debug=True)
    print(test)
