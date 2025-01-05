import onnxruntime
import numpy as np
import cv2
from photolink.models.scrfd import run_scrfd_inference
from photolink.utils.image_loader import ImageLoader
from loguru import logger
import copy

def extract_5_keypoints(landmarks_3d):
    """
    Convert 468 face mesh landmarks to 5 keypoints:
    1) Left eye center
    2) Right eye center
    3) Nose tip
    4) Left mouth corner
    5) Right mouth corner

    Parameters
    ----------
    landmarks_3d : np.ndarray of shape (468, 3)
        Full set of face mesh points, each is (x, y, z).

    Returns
    -------
    np.ndarray of shape (5, 3)
        The 3D coordinates for these 5 key facial points.
        If you only want 2D, slice [:, :2].
    """

    # ---- Example Indices (MediaPipe Face Mesh) ----
    # Each list here contains multiple indices around that region. We'll average them.
    # You can adjust these indices for your model. 
    LEFT_EYE_IDXS = [33, 133, 160, 158, 159, 144, 153, 145]  # example region around left eye
    RIGHT_EYE_IDXS = [263, 362, 387, 385, 386, 373, 380, 374]
    
    # Nose tip is often around index 1 or 4 in MediaPipe. 
    # We'll pick 4 here, but you can check visually if you prefer index 1 or 2, etc.
    NOSE_TIP_IDX = 4

    # Mouth corners (approx.) for MediaPipe FaceMesh
    LEFT_MOUTH_CORNER_IDX = 61
    RIGHT_MOUTH_CORNER_IDX = 291

    # ---- 1) Get the (x, y, z) for each region. ----
    left_eye_points = landmarks_3d[LEFT_EYE_IDXS]     # shape (N, 3)
    right_eye_points = landmarks_3d[RIGHT_EYE_IDXS]
    nose_tip_point = landmarks_3d[NOSE_TIP_IDX]       # shape (3,)
    left_mouth_corner_point = landmarks_3d[LEFT_MOUTH_CORNER_IDX]
    right_mouth_corner_point = landmarks_3d[RIGHT_MOUTH_CORNER_IDX]

    # ---- 2) Compute average for each eye region. ----
    left_eye_center = np.mean(left_eye_points, axis=0)   # shape (3,)
    right_eye_center = np.mean(right_eye_points, axis=0)

    # ---- 3) Construct final array of shape (5, 3). ----
    # If you only need 2D, you can do `.reshape(5,3)[:, :2]` later.
    keypoints_3d = np.stack([
        left_eye_center,
        right_eye_center,
        nose_tip_point,
        left_mouth_corner_point,
        right_mouth_corner_point,
    ], axis=0)

    return keypoints_3d


def visualize_landmarks(image, landmarks, color=(0,255,0), radius=1, thickness=-1):
    """
    Draw 2D circles for each (x, y, z) in `landmarks` onto `image`.

    image: np.ndarray (H,W,3) in BGR
    landmarks: np.ndarray of shape (468, 3) => (x, y, z)
    color: BGR tuple for the circle
    radius: circle radius in pixels
    thickness: -1 for filled circle
    """
    image = copy.deepcopy(image)
    for (lx, ly, lz) in landmarks:
        cv2.circle(
            image, 
            (int(lx), int(ly)), 
            radius, 
            color, 
            thickness
        )
    
    return image

if __name__ == '__main__':
    # 1) Load your test image
    im_loader = ImageLoader("sample/IMG_0066.JPG")
    ds_image = np.array(im_loader.get_downsampled_image())  # (H,W,3) BGR

    # 2) Detect face with SCRFD
    detection_result = run_scrfd_inference(ds_image, heuristic_filter=True)
    face_bounding_box = detection_result['faces'].tolist()[0]  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(x) for x in face_bounding_box]
    crop_width = x2 - x1
    crop_height = y2 - y1
    logger.info(f'Detected face bounding box: {face_bounding_box}')

    # 3) Crop + resize to 192x192
    cropped_face = ds_image[y1:y2, x1:x2, :]
    resized_face = cv2.resize(cropped_face, (192, 192))
    resized_face = resized_face[..., ::-1] / 255.0  # BGR->RGB, scale [0..1]
    resized_face = resized_face.astype(np.float32)  # (192,192,3)
    resized_face = np.transpose(resized_face, (2, 0, 1))  # => (3,192,192)
    resized_face = np.expand_dims(resized_face, axis=0)   # => (1,3,192,192)

    # 4) Load FaceMesh ONNX
    face_mesh_model = r'C:\Users\choph\photomatcher\assets\weights\face_mesh\face_mesh_Nx3x192x192_post.onnx'
    session_option_facemesh = onnxruntime.SessionOptions()
    session_option_facemesh.log_severity_level = 3
    face_mesh_sess = onnxruntime.InferenceSession(
        face_mesh_model,
        sess_options=session_option_facemesh,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CPUExecutionProvider',
        ],
    )
    face_mesh_input_names = [inp.name for inp in face_mesh_sess.get_inputs()]
    face_mesh_output_names = [out.name for out in face_mesh_sess.get_outputs()]

    # 5) Prepare bounding box data
    np_crop_x1 = np.asarray(x1, dtype=np.int32).reshape(-1,1)
    np_crop_y1 = np.asarray(y1, dtype=np.int32).reshape(-1,1)
    np_crop_width = np.asarray(crop_width, dtype=np.int32).reshape(-1,1)
    np_crop_height = np.asarray(crop_height, dtype=np.int32).reshape(-1,1)

    logger.info('Running face mesh inference...')
    scores, final_landmarks = face_mesh_sess.run(
        output_names=face_mesh_output_names,
        input_feed={
            face_mesh_input_names[0]: resized_face,   # (1,3,192,192)
            face_mesh_input_names[1]: np_crop_x1,     # (1,1)
            face_mesh_input_names[2]: np_crop_y1,     # (1,1)
            face_mesh_input_names[3]: np_crop_width,  # (1,1)
            face_mesh_input_names[4]: np_crop_height, # (1,1)
        }
    )

    # final_landmarks shape => (1, 468, 3)
    # scores shape => (1, )
    conf = scores[0]
    if conf > 0.95:
        landmark_3d_array = final_landmarks[0]  # shape (468, 3)

        # Convert 468 => 5
        five_keypoints_3d = extract_5_keypoints(landmark_3d_array)
        
        # If EdgeFace only needs 2D, do:
        # five_keypoints_2d = five_keypoints_3d[:, :2]
        
        logger.info(f"5 keypoints (3D): {five_keypoints_3d}")
        
        # If you want to visualize them on `ds_image`, do:
        # (Note: converting 3D -> 2D by ignoring Z)
        five_keypoints_2d = five_keypoints_3d[:, :2].astype(int)
        for (px, py) in five_keypoints_2d:
            cv2.circle(ds_image, (px, py), 4, (0, 0, 255), -1)
        
        figure = visualize_landmarks(ds_image, five_keypoints_3d, color=(0,255,0))
        cv2.imwrite('output.jpg', figure)
    else:
        logger.warning(f'Low FaceMesh confidence: {conf}')

    # TODO: Convert these to singleton pattern. Add face mesh and 5 keypoints to dict and return it.
    # Complete the edgeface.py file conversion. 