import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import IGNORE_INDEX
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list      = []
    images_list          = []
    images_clip_list     = []
    conversation_list    = []
    masks_list           = []
    label_list           = []
    resize_list          = []
    questions_list       = []
    sampled_classes_list = []
    offset_list          = [0]
    cnt                  = 0
    inferences           = []

    for (
        image_path, images, images_clip, conversations,
        masks, label, resize, questions, sampled_classes, inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv    = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    sep     = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds    = conversation.split(conv.sep2)
        cur_len   = 1
        target[:cur_len] = IGNORE_INDEX

        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len       = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len       = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    # truncate long sequences during training (not inference)
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            input_ids       = input_ids[:, :truncate_len]
            targets         = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths":         image_path_list,
        "images":              torch.stack(images_list, dim=0),
        "images_clip":         torch.stack(images_clip_list, dim=0),
        "input_ids":           input_ids,
        "labels":              targets,
        "attention_masks":     attention_masks,
        "masks_list":          masks_list,
        "label_list":          label_list,
        "resize_list":         resize_list,
        "offset":              torch.LongTensor(offset_list),
        "questions_list":      questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference":           inferences[0],
        "conversation_list":   conversation_list,
    }


class MedDataset(torch.utils.data.Dataset):
    """
    Medical segmentation + VQA dataset for LISA fine-tuning.

    Loads from one or more JSON files (produced by build_dataset.py).
    Each sample has: image, mask, type, question, answer.

    Image / mask paths in the JSON are relative to `base_dir`
    (e.g. "BUSI/Dataset_BUSI_with_GT/benign/foo.png" + base_dir="./dataset").

    Conversation format fed to LISA:
        User : <image>\\n {question}
        Model: {answer} [SEG].

    `inference=False`  → training  (collate_fn truncates long sequences)
    `inference=True`   → val/test  (no truncation, full sequence kept)
    """

    pixel_mean   = torch.Tensor([123.675, 116.28,  103.53]).view(-1, 1, 1)
    pixel_std    = torch.Tensor([58.395,  57.12,   57.375]).view(-1, 1, 1)
    img_size     = 1024
    ignore_label = 255

    def __init__(
        self,
        json_paths,          # list[str] — paths to dataset JSON files
        tokenizer,
        vision_tower,        # str — e.g. "openai/clip-vit-large-patch14"
        base_dir="./dataset", # prepended to relative image/mask paths in JSON
        image_size=1024,
        inference=False,
    ):
        self.base_dir    = base_dir
        self.image_size  = image_size
        self.inference   = inference
        self.tokenizer   = tokenizer

        self.samples = []
        for path in json_paths:
            with open(path) as f:
                data = json.load(f)
            self.samples.extend(s for s in data if s.get("type") == "baseline")

        self.transform           = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.samples)

    def _resolve(self, rel_path: str) -> str:
        """Join base_dir with a relative path from the JSON."""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.base_dir, rel_path)

    def _build_conversation(self, question: str, answer: str):
        """Build a single-turn conversation: user asks, model answers + [SEG]."""
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], answer.strip() + " [SEG].")
        return [conv.get_prompt()]

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and pad to img_size × img_size."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    def __getitem__(self, idx):
        item       = self.samples[idx]
        image_path = self._resolve(item["image"])
        mask_path  = self._resolve(item["mask"])
        question   = item["question"]
        answer     = item["answer"]

        # --- load image ---
        image_np = cv2.imread(image_path)
        if image_np is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # --- load mask → binary [1, H, W] ---
        # Read as BGR so color masks (e.g. BKAI) are handled correctly:
        # black (0,0,0) → 0, any other color → 1.
        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        # threshold > 10 guards against JPEG compression artifacts near edges
        mask_binary = (mask_bgr.sum(axis=2) > 10).astype(np.uint8)
        masks = torch.from_numpy(mask_binary).unsqueeze(0)

        # --- conversation ---
        conversations = self._build_conversation(question, answer)

        # --- CLIP preprocessing (224×224 patch encoder input) ---
        image_clip = self.clip_image_processor.preprocess(
            image_np, return_tensors="pt"
        )["pixel_values"][0]

        # --- SAM preprocessing (resize longest side → pad to 1024×1024) ---
        image_sam = self.transform.apply_image(image_np)
        resize    = image_sam.shape[:2]
        image_sam = self.preprocess(
            torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()
        )

        # ignore_label canvas — same convention as original LISA val datasets
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,    # str
            image_sam,     # [3, 1024, 1024]  float  — SAM input
            image_clip,    # [3, 224, 224]    float  — CLIP input
            conversations, # list[str]        — formatted prompt
            masks,         # [1, H, W]        uint8  — ground-truth mask
            labels,        # [H, W]           float  — ignore canvas
            resize,        # (h, w) after SAM resize, before padding
            answer,        # str  — ground-truth answer; surfaced via questions_list in batch
            None,          # sampled_classes  — unused, kept for collate_fn compat
            self.inference,
        )
