import photomatcher.model.yunet as yunet
import photomatcher.model.sface as sface
import os   
import shutil

def run_ml_model(image_path, fail_path):
    """Process face detection and recognition for images"""
    detection_result = yunet.run_face_detection(image_path)

    failed_image = os.path.join(fail_path, os.path.basename(image_path))

    if "error" in detection_result:
        shutil.copy(image_path, failed_image)
        return f"Face detection error on source image {image_path}: {detection_result['error']}", None

    image = detection_result["image"]
    embedding_dict = sface.run_face_recognition(image, detection_result["faces"])

    if "error" in embedding_dict:
        shutil.copy(image_path, failed_image)
        return f"Face recognition error on source image {image_path}: {embedding_dict['error']}", None
    
    embedding = embedding_dict["embeddings"]

    return None, embedding
