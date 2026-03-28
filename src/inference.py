"""Model loading and inference utilities for Qwen2.5-VL.

Handles 4-bit NF4 quantization, optional LoRA adapter loading,
and the two-step processor pattern required by Qwen2.5-VL.
"""

import json
import re

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from .prompts import SYSTEM_PROMPT, USER_PROMPT


def load_model(model_id, adapter_id=None, quantize_4bit=True):
    """Load Qwen2.5-VL with optional LoRA adapter and 4-bit quantization.

    Returns (model, processor) tuple. The model is set to eval mode.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)

    if adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_id)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    return model, processor


def run_inference(model, processor, image, system_prompt=None):
    """Run single-image inference using the Qwen2.5-VL two-step processor pattern.

    Args:
        model: Loaded Qwen2.5-VL model.
        processor: Corresponding AutoProcessor.
        image: PIL Image or file path.
        system_prompt: Override default system prompt.

    Returns:
        Raw text output from the model (unparsed).
    """
    if isinstance(image, (str, bytes)):
        image = Image.open(image).convert("RGB")

    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=img_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    gen_trimmed = gen[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]


def extract_json(text):
    """Strip markdown code fences and parse JSON from model output."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    cleaned = m.group(1).strip() if m else text.strip()
    return json.loads(cleaned)


def validate_schema(raw_text):
    """Check if raw prediction is valid JSON with required schema fields.

    Returns (parsed_obj, error_string). error_string is None on success.
    """
    try:
        obj = extract_json(raw_text)
    except (json.JSONDecodeError, ValueError):
        return None, "invalid_json"

    errors = []
    for field in ["caption", "scene_type", "objects", "infrastructure", "terrain"]:
        if field not in obj:
            errors.append(f"missing_{field}")

    if errors:
        return obj, ", ".join(errors)
    return obj, None
