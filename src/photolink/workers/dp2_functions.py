"""Run yolov11 and kosmos together to make comparison"""

import cv2
import numpy as np

from PIL import Image
import IPython
from photolink.utils.function import safe_load_image


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def draw_entity_boxes_on_image(image, entities):
    """_summary_
    Args:
        image (_type_): image or image path
        collect_entity_location (_type_): _description_
    """
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    if len(entities) == 0:
        return image

    new_image = image.copy()
    previous_bboxes = []
    # size of text
    text_size = 1
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 3
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    for entity_name, (start, end), bboxes in entities:
        for x1_norm, y1_norm, x2_norm, y2_norm in bboxes:
            orig_x1, orig_y1, orig_x2, orig_y2 = (
                int(x1_norm * image_w),
                int(y1_norm * image_h),
                int(x2_norm * image_w),
                int(y2_norm * image_h),
            )
            # draw bbox
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

            x1 = orig_x1 - l_o
            y1 = orig_y1 - l_o

            if y1 < text_height + text_offset_original + 2 * text_spaces:
                y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                x1 = orig_x1 + r_o

            # add text background
            (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = (
                x1,
                y1 - (text_height + text_offset_original + 2 * text_spaces),
                x1 + text_width,
                y1,
            )

            for prev_bbox in previous_bboxes:
                while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                    text_bg_y1 += text_height + text_offset_original + 2 * text_spaces
                    text_bg_y2 += text_height + text_offset_original + 2 * text_spaces
                    y1 += text_height + text_offset_original + 2 * text_spaces

                    if text_bg_y2 >= image_h:
                        text_bg_y1 = max(
                            0,
                            image_h - (text_height + text_offset_original + 2 * text_spaces),
                        )
                        text_bg_y2 = image_h
                        y1 = image_h
                        break

            alpha = 0.5
            for i in range(text_bg_y1, text_bg_y2):
                for j in range(text_bg_x1, text_bg_x2):
                    if i < image_h and j < image_w:
                        if j < text_bg_x1 + 1.35 * c_width:
                            # original color
                            bg_color = color
                        else:
                            # white
                            bg_color = [255, 255, 255]
                        new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

            cv2.putText(
                new_image,
                f"  {entity_name}",
                (x1, y1 - text_offset_original - 1 * text_spaces),
                cv2.FONT_HERSHEY_COMPLEX,
                text_size,
                (0, 0, 0),
                text_line,
                cv2.LINE_AA,
            )
            # previous_locations.append((x1, y1))
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

    pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])

    return pil_image

if __name__ == "__main__":

    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import time

    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    print('loaded model and processor')
    
    start_time = time.time()

    prompt = "<grounding> where is the student?"  # <grounding> is used to prompt the model to generate locations tokens
        
    image = safe_load_image("sample/BCITCS24-C4P1-0008.JPG", return_numpy=False)

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Specify `cleanup_and_extract=False` in order to see the raw model generation.
    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    print(f"Raw model generation: {processed_text}")

    processed_text, entities = processor.post_process_generation(generated_text)
    end_time = time.time()
    print(f"Time taken: {end_time-start_time}")
    print(f"Cleaned up generated text: {processed_text=}")

    print(f"Extracted entities: {entities}")


    new_image = draw_entity_boxes_on_image(image, entities)
    new_image.save('test.png')
