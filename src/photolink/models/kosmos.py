"""Run Kosmos inference."""
# export script.
from pathlib import Path
import gc

import torch
import openvino as ov
from transformers import AutoProcessor, AutoModelForVision2Seq
from photolink.utils.function import safe_load_image
from torch import nn
from typing import Optional, List
from transformers.models.kosmos2.modeling_kosmos2 import (
    create_position_ids_from_input_ids,
)

from transformers.generation import GenerationConfig, GenerationMixin
from transformers.models.kosmos2.modeling_kosmos2 import (
    Kosmos2ForConditionalGenerationModelOutput,
)

import numpy as np

class WraperInternalVisionModel:
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    post_layernorm = model.vision_model.model.post_layernorm

class VisionModelWrapper(torch.nn.Module):
    def __init__(self, model_ir_path):
        super().__init__()
        self.model = WraperInternalVisionModel()
        self.vision_model = core.compile_model(model_ir_path, device.value)

    def forward(self, pixel_values, **kwargs):
        vision_model_output = self.vision_model(pixel_values)[0]

        return [torch.from_numpy(vision_model_output)]


class ImageToTextProjectionModelWrapper(torch.nn.Module):
    def __init__(self, model_ir_path):
        super().__init__()
        self.image_to_text_projection = core.compile_model(model_ir_path, device.value)

    def forward(self, image_embeds):
        output = self.image_to_text_projection(image_embeds)
        image_embeds = output[0]
        projection_attentions = output[1]
        return image_embeds, projection_attentions



class KosmosForCausalLMWrapper(GenerationMixin):
    def __init__(self, first_stage_model_path, second_stage_model_path, device):
        self.model_stage_1 = core.compile_model(first_stage_model_path, device.value)
        self.model_stage_2 = core.read_model(second_stage_model_path)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model_stage_2.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model_stage_2.outputs)}
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        self.key_value_output_names = [key for key in self.output_names if "present" in key]
        self.model_stage_2 = core.compile_model(self.model_stage_2, device.value)

        self.request = self.model_stage_2.create_infer_request()
        self.config = model.config
        self.generation_config = GenerationConfig.from_model_config(model.config)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self.lm_head = nn.Linear(
            in_features=model.text_model.config.embed_dim,
            out_features=model.text_model.config.vocab_size,
            bias=False,
        )
        self._supports_cache_class = False

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ):
        return self.forward(
            input_ids,
            attention_mask,
            image_embeds,
            image_embeds_position_mask,
            position_ids,
            past_key_values,
        )

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ):
        if past_key_values is None:
            outs = self.model_stage_1(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "image_embeds": image_embeds,
                    "image_embeds_position_mask": image_embeds_position_mask,
                }
            )
            lm_logits = model.text_model.lm_head(torch.from_numpy(outs[0]))

            pkv = list(outs.values())[1:]
            pkv = tuple(pkv[i : i + 2] for i in range(0, len(pkv), 2))

            return Kosmos2ForConditionalGenerationModelOutput(logits=lm_logits, past_key_values=pkv)

        if past_key_values is not None:
            past_key_values = tuple(past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer)
            inputs_ = {
                "input_ids": input_ids[:, -1].unsqueeze(-1),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            inputs_.update(dict(zip(self.key_value_input_names, past_key_values)))

        # Run inference
        self.request.start_async(inputs_, share_inputs=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor("logits").data)
        logits = model.text_model.lm_head(logits)

        # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
        past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
        # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)

        past_key_values = tuple(past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))

        return Kosmos2ForConditionalGenerationModelOutput(logits=logits, past_key_values=past_key_values)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        image_embeds=None,
        image_embeds_position_mask=None,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        position_ids = None

        # cut input_ids if past_key_values is used
        if past_key_values is not None:
            position_ids = create_position_ids_from_input_ids(
                input_ids,
                padding_idx=model.text_model.config.pad_token_id,
                past_key_values_length=0,
            )[:, -1:]

            input_ids = input_ids[:, -1:]
            image_embeds = None
            image_embeds_position_mask = None
        elif image_embeds_position_mask is not None:
            batch_size, seq_len = input_ids.size()
            mask_len = image_embeds_position_mask.size()[-1]
            image_embeds_position_mask = torch.cat(
                (
                    image_embeds_position_mask,
                    torch.zeros(
                        size=(batch_size, seq_len - mask_len),
                        dtype=torch.bool,
                        device=input_ids.device,
                    ),
                ),
                dim=1,
            )

        return {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "image_embeds_position_mask": image_embeds_position_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    @staticmethod
    # Copied from transformers.models.umt5.modeling_umt5.UMT5ForConditionalGeneration._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),)
        return reordered_past


class Kosmos2ForConditionalGenerationWrapper:
    def __init__(
        self,
        vision_model_path,
        image_to_text_projection_model_path,
        first_stage_model_path,
        second_stage_model_path,
        device,
    ):
        self.vision_model = VisionModelWrapper(vision_model_path)
        self.image_to_text_projection = ImageToTextProjectionModelWrapper(image_to_text_projection_model_path)
        self.text_model = KosmosForCausalLMWrapper(first_stage_model_path, second_stage_model_path, device)

    def generate(
        self,
        pixel_values=None,
        image_embeds_position_mask=None,
        input_ids=None,
        attention_mask=None,
        image_embeds=None,
        **kwargs,
    ):
        vision_model_output = self.vision_model(pixel_values)
        image_embeds = model.vision_model.model.post_layernorm(vision_model_output[0])
        # normalized features
        image_embeds = nn.functional.normalize(image_embeds, dim=-1)
        image_embeds, projection_attentions = self.image_to_text_projection(image_embeds.detach().numpy())

        output = self.text_model.generate(
            input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            **kwargs,
        )

        return output

def generate_entities(model, processor, inputs):
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        max_new_tokens=128,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Specify `cleanup_and_extract=False` in order to see the raw model generation.
    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    print(f"Raw model generation: {processed_text}")

    # By default, the generated  text is cleanup and the entities are extracted.
    processed_text, entities = processor.post_process_generation(generated_text)

    print(f"Cleaned up generated text: {processed_text=}")
    # `An image of a snowman warming himself by a fire.`

    print(f"Extracted entities: {entities}")
    return entities


if __name__ == "__main__":

    import IPython as ip
    from photolink.models.notebook_utils import device_widget, quantization_widget, collect_calibration_data
    import time
    import nncf

    models_base_folder = Path("assets/weights")

    # fp16
    VISION_MODEL_IR_PATH = models_base_folder / "vision_model.xml"
    IMAGE_TO_TEXT_PROJECTION_MODEL_IR_PATH = models_base_folder / "image_to_text_projection_model.xml"
    FIRST_STAGE_MODEL_PATH = models_base_folder / "kosmos_input_embed.xml"
    SECOND_STAGE_MODEL_PATH = models_base_folder / "kosmos_with_past.xml"

    # load the core model and processor
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    # load the wrapper 
    core = ov.Core()
    device = device_widget()

    ov_model = Kosmos2ForConditionalGenerationWrapper(
    VISION_MODEL_IR_PATH,
    IMAGE_TO_TEXT_PROJECTION_MODEL_IR_PATH,
    FIRST_STAGE_MODEL_PATH,
    SECOND_STAGE_MODEL_PATH,
    device,
)
    # load the inputs
    image = safe_load_image("sample/BCITCS24-C4P1-0008.JPG", return_numpy=False)
    prompt_main = "<grounding> where is the student?"  # <grounding> is used to prompt the model to generate locations tokens
    inputs = processor(text=prompt_main, images=image, return_tensors="pt")

    # call the inference on the wrapper
    start_time = time.time()
    response = generate_entities(ov_model, processor, inputs)
    end_time = time.time()
    print(f"Befire quantizing, time taken: {end_time-start_time}")

