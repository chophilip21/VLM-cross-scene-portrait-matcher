import cv2
import numpy as np
import onnxruntime as ort

class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class Colors:
    """Color palette for visualization."""

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


class YOLOSeg:
    """YOLOv8 style segmentation model."""

    def __init__(
        self, onnx_model, human_only=False, num_candidates=-1, heuristic_threshold=0.5
    ):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
            human_only (bool): If true, only detects humans (class index 0).
            num_candidates (int): Number of candidates to retain. Set to -1 for no filtering.
            heuristic_threshold (float): Threshold to filter out poor heuristic scores.
        """
        self.human_only = human_only
        self.num_candidates = num_candidates
        self.heuristic_threshold = heuristic_threshold

        if not self.human_only and self.num_candidates != -1:
            raise ValueError(
                "num_candidates filtering is only available when human_only is set to True."
            )

        # Build Ort session
        self.session = ort.InferenceSession(
            onnx_model,
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if ort.get_device() == "GPU"
                else ["CPUExecutionProvider"]
            ),
        )

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = (
            np.half
            if self.session.get_inputs()[0].type == "tensor(float16)"
            else np.single
        )

        # Get model width and height
        self.model_height, self.model_width = [
            x.shape for x in self.session.get_inputs()
        ][0][-2:]

        # Load COCO class names
        self.classes = class_names

        # Create color palette
        self.color_palette = Colors()

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )

        return boxes, segments, masks

    def preprocess(self, img):
        # Resize and pad input image using letterbox()
        shape = img.shape[:2]
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (
            new_shape[0] - new_unpad[1]
        ) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        img = (
            np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype)
            / 255.0
        )
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(
        self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32
    ):
        x, protos = preds[0], preds[1]

        # Transpose dim 1: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Combine boxes, scores, and classes into one matrix
        x = np.c_[
            x[..., :4],
            np.amax(x[..., 4:-nm], axis=-1),
            np.argmax(x[..., 4:-nm], axis=-1),
            x[..., -nm:],
        ]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Filter based on human_only flag
        if self.human_only:
            x = x[x[:, 5] == 0]  # Only keep class index 0 (human)

        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescale boxes
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            if self.human_only and self.num_candidates != -1:
                x = self.heuristics_filter(x, im0.shape)

            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
            segments = self.masks2segments(masks)

            return x[..., :6], segments, masks
        else:
            return [], [], []

    def heuristics_filter(self, boxes, im_shape):
        """
        Apply heuristic filtering based on size and distance from the center, with a heuristic threshold.

        Candidates with heuristic scores below the threshold are removed unless no candidates pass the threshold,
        in which case the top candidate will be returned.

        Args:
            boxes (numpy.ndarray): Detected bounding boxes with scores and class ids.
            im_shape (tuple): Shape of the input image (h, w).

        Returns:
            filtered_boxes (numpy.ndarray): Filtered bounding boxes.
        """
        img_center = np.array([im_shape[1] / 2, im_shape[0] / 2])

        # Calculate the size of each bounding box
        box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Calculate heuristic scores (normalized between 0 and 1)
        max_size = np.max(box_sizes)
        # Correct max_dist calculation
        max_dist = np.linalg.norm(np.array([im_shape[1] / 2, im_shape[0] / 2]))

        scores = []
        for i in range(len(boxes)):
            box_center = np.array(
                [(boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2]
            )
            size_penalty = (
                box_sizes[i] / max_size if max_size > 0 else 0
            )  # Avoid division by zero
            dist_from_center = np.linalg.norm(box_center - img_center)

            # Normalize distance penalty correctly
            dist_penalty = 1 - (
                dist_from_center / max_dist
            )  # Normalize distance penalty to [0,1]

            score = 0.5 * dist_penalty + 0.5 * size_penalty  # Weighted combination
            scores.append(score)

        scores = np.array(scores)

        # Sort boxes based on scores in descending order
        sorted_score_indices = np.argsort(-scores)
        boxes = boxes[sorted_score_indices]
        scores = scores[sorted_score_indices]

        # Filter based on heuristic threshold
        valid_indices = scores >= self.heuristic_threshold
        if np.any(valid_indices):
            # Select all valid candidates sorted by score
            filtered_boxes = boxes[valid_indices]
        else:
            # No candidates pass the threshold; return the top candidate
            filtered_boxes = boxes[:1]

        # If there are more than num_candidates, select the top ones
        if len(filtered_boxes) > self.num_candidates:
            filtered_boxes = filtered_boxes[: self.num_candidates]

        return filtered_boxes

    @staticmethod
    def masks2segments(masks):
        """
        Takes a list of masks(n,h,w) and returns a list of segments(n,xy).

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        Takes a mask and a bounding box, and returns a mask that is cropped to the bounding box, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
        quality but is slower, from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = (
            np.matmul(masks_in, protos.reshape((c, -1)))
            .reshape((-1, mh, mw))
            .transpose(1, 2, 0)
        )  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(
            masks, im0_shape
        )  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(
                im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1]
            )  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (
                im1_shape[0] - im0_shape[0] * gain
            ) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(
            round(im1_shape[1] - pad[0] + 0.1)
        )
        if len(masks.shape) < 2:
            raise ValueError(
                f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
            )
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """
        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            cls_ = int(cls_)

            # draw contour and fill mask
            cv2.polylines(
                im, np.int32([segment]), True, (255, 255, 255), 2
            )  # white borderline
            cv2.fillPoly(
                im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True)
            )

            # draw bbox rectangle
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # Show image
        if vis:
            cv2.imshow("demo", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image
        if save:
            cv2.imwrite("test/demo.jpg", im)


if __name__ == "__main__":
    from pathlib import Path
    import time
    from loguru import logger

    model_path = str(
        Path(r"C:\Users\choph\photomatcher\assets\weights\yolo11m-seg.onnx")
    )
    img_url = str(
        Path(r"C:\Users\choph\photomatcher\demo\UCALCF23-C1-AWARDS-00012.jpg")
    )
    conf = 0.5
    iou = 0.45
    model = YOLOSeg(
        model_path, human_only=True, num_candidates=3, heuristic_threshold=0.7
    )
    # model = YOLOSeg(model_path, human_only=False)

    logger.info(f"Model loaded from {model_path}")

    # Read image by OpenCV
    img = cv2.imread(img_url)

    # Inference
    start = time.time()
    boxes, segments, _ = model(img, conf_threshold=conf, iou_threshold=iou)
    logger.info(f"Time cost: {time.time() - start:.3f}s")

    # Draw bboxes and polygons
    if len(boxes) > 0:
        model.draw_and_visualize(img, boxes, segments, vis=False, save=True)
