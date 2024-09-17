"""Add lower level dp2 functions to avoid cluttering the main functional modules."""

from photolink import get_application_path, get_config
from pathlib import Path
import os
from photolink.models.yoloworld import YOLOWorld, read_class_embeddings
from photolink.utils.function import safe_load_image, search_all_images
from loguru import logger
import numpy as np
import cv2

def run_world_detection(
    conf_thres=0.001,
    iou_thres=0.5,
    image_list: list = None,
    debug: bool = False,
) -> dict:
    """Use yoloworld to detect humans based on precomputed embeddings.

    Iterate over list of images, and return dict based on the image name as key.
    """

    application_path = get_application_path()
    config = get_config()
    model_path = str(application_path / Path(config.get("YOLO-WORLD", "LOCAL_PATH")))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    embed_path = str(
        application_path / Path(config.get("YOLO-WORLD", "GRAD_EMBEDDING_LOCAL"))
    )

    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding path not found: {embed_path}")

    # Initialize YOLO-World object detector
    yoloworld_detector = YOLOWorld(
        model_path, conf_thres=conf_thres, iou_thres=iou_thres
    )
    class_embeddings, _ = read_class_embeddings(embed_path)
    output_dict = {}

    # iterate over images.
    for image_path in image_list:

        try:
            img = safe_load_image(image_path)
        except Exception as e:
            print(f"Error loading image: {image_path}, {e}")
            continue

        boxes, scores, _ = yoloworld_detector(img, class_embeddings)

        indices = np.argsort(scores)[::-1]
        indices = indices[:2]
        output_dict[image_path] = {}

        for i in indices:
            box = boxes[i]
            score = scores[i]

            output_dict[image_path][i] = {
                "box": box,
                "score": score,
            }

            # debug draw boxes, and index label as text.
            if debug:

                # draw box
                cv2.rectangle(
                    img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0),
                    2,
                )

                # draw text of score.
                cv2.putText(
                    img,
                    f"{score*100:.2f}%",
                    (int(box[0]), int(box[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 0),
                    5,
                )

        if debug:
            debug_folder = "debug"
            os.makedirs(debug_folder, exist_ok=True)
            stem = os.path.basename(image_path).split(".")[0]
            save_name = f"{stem}_detected.jpg"
            save_name = os.path.join(debug_folder, save_name)   
            logger.info(f"Saving debug image: {save_name}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_name, img)


        logger.info(f"Detected {len(boxes)} humans in {image_path}")

    return output_dict


if __name__ == "__main__":

    demo_path = str(Path(r"C:\Users\choph\philip\FOR PHIL\USFU\C1\C1P1\EDITS"))
    img_list = search_all_images(demo_path)
    test = run_world_detection(image_list=img_list, debug=True)
    print(test)
