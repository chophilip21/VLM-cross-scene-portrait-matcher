from ov_florence2_helper import convert_florence2, get_model_selector, OVFlorence2Model
from pathlib import Path
from notebook_utils import device_widget
import requests
from PIL import Image
from transformers import AutoProcessor
import copy
from PIL import Image, ImageOps
from io import BytesIO
from typing import Union
import numpy as np


def _copy_image_meta(src: Image.Image, dest: Image.Image):
    preserve_metadata_keys = ["info", "icc_profile", "exif", "dpi", "applist", "format"]
    for key in preserve_metadata_keys:
        if hasattr(src, key):
            setattr(dest, key, copy.deepcopy(getattr(src, key)))
    return dest


def safe_load_image(image: Union[bytes, str], return_numpy=True) -> np.ndarray:
    """Load an image from bytes or a file path, and ensure the orientation is correct."""
    # make sure image is bytes or a valid file path
    if isinstance(image, str):
        with open(image, "rb") as f:
            image = f.read()
    elif not isinstance(image, bytes):
        raise TypeError(f"image must be bytes or a file path, not {type(image)}")
    pil_image = Image.open(BytesIO(image))

    # Make sure the orientation is correct
    if hasattr(pil_image, "_getexif") and pil_image._getexif() is not None:
        new_pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = _copy_image_meta(pil_image, new_pil_image)

    if return_numpy:
        return np.array(pil_image)

    return pil_image


if __name__ == "__main__":
    import time
    from gradio_helper import plot_bbox
    import IPython
    from notebook_utils import quantization_widget


    model_selector = get_model_selector()
    model_id = model_selector.value
    model_path = Path(model_id.split("/")[-1])

    convert_florence2(model_id, model_path)
    print('conversion success...')
    device = device_widget()
    print(f"Device : {device.value}")

    model = OVFlorence2Model(model_path, device.value)
    print("FP32 openvino model loaded...")


    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print('test')


    image_path = "/Users/philipcho/photomatcher/sample/BCITCS24-C4P1-0004.JPG"
    image = safe_load_image(image_path, return_numpy=False)
    prompt = "<CAPTION_TO_PHRASE_GROUNDING>The main graduating student"

    # start = time.time()
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    IPython.embed()

    # generated_ids = model.generate(
    #     input_ids=inputs["input_ids"],
    #     pixel_values=inputs["pixel_values"],
    #     max_new_tokens=1024,
    #     do_sample=False,
    #     num_beams=3,
    #     output_scores=True,
    # )

    # # IPython.embed()

    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # parsed_answer = processor.post_process_generation(
    #     generated_text, task="<OD>", image_size=(image.width, image.height)
    # )

    # print(f"Predicted: {parsed_answer}, Inference time: {time.time() - start:.2f}s")

    # # save figure
    # fig = plot_bbox(image, parsed_answer["<OD>"])
    # fig.savefig("output.png")
