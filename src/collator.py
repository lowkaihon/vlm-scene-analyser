"""Custom data collator for Qwen2.5-VL fine-tuning.

Implements the two-step processor pattern required by Qwen2.5-VL:
build messages -> chat template -> process_vision_info -> processor.
Masks prompt tokens in labels so loss is computed only on the assistant response.
"""

from PIL import Image
from qwen_vl_utils import process_vision_info


class MultimodalCollator:
    """Collator for Qwen2.5-VL multimodal fine-tuning.

    Args:
        processor: AutoProcessor for Qwen2.5-VL.
        system_prompt: System prompt string.
        user_prompt: User prompt string.
        max_seq_length: Maximum sequence length for truncation.
        augment_fn: Optional callable (PIL.Image -> PIL.Image) for augmentation.
            Pass augment_image for training, None for validation.
    """

    def __init__(self, processor, system_prompt, user_prompt, max_seq_length, augment_fn=None):
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_seq_length = max_seq_length
        self.augment_fn = augment_fn

    def __call__(self, examples):
        texts = []
        prompt_lens = []
        all_images = []

        for ex in examples:
            image = Image.open(ex["image_path"]).convert("RGB")
            if self.augment_fn is not None:
                image = self.augment_fn(image)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
                {"role": "assistant", "content": ex["target_json"]},
            ]

            # Full text (with assistant response)
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # Prompt-only text (system + user, no assistant)
            prompt_text = self.processor.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True
            )
            texts.append(full_text)

            image_inputs, _ = process_vision_info(messages)
            if image_inputs:
                all_images.extend(image_inputs)

            # Compute prompt token length (includes expanded image tokens)
            prompt_batch = self.processor(
                text=[prompt_text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
            )
            prompt_lens.append(prompt_batch["input_ids"].shape[1])

        batch = self.processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Labels: mask prompt + padding with -100 (only compute loss on assistant response)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        batch["labels"] = labels

        return batch
