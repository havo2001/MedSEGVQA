# MedSegVQA


## Dataset format

Every `train.json` / `val.json` / `test.json` is a list of samples:

```json
[
  {
    "image":    "BUSI/Dataset_BUSI_with_GT/benign/benign (1).png",
    "mask":     "BUSI/Dataset_BUSI_with_GT/benign/benign (1)_mask.png",
    "type":     "baseline",
    "question": "What can you see in the image?",
    "answer":   "A small oval-shaped benign tumor in the breast ultrasound image."
  }
]
```

Paths are relative to `--dataset_dir` (default `./dataset`).
Masks can be grayscale or color (e.g. BKAI red-on-black) — any non-black pixel is treated as foreground.

---

## Training

After each epoch the model is automatically validated on **BUSI, ISIC, Kvasir-SEG, BUID** (val split).
The best checkpoint is saved based on average gIoU across these 4 datasets.

```bash
# Remove --val_dataset and --val_split if you want to run validation separately
!deepspeed --master_port=24999 train_ds.py \
--version="xinlai/LISA-13B-llama2-v1" \
--dataset_dir="./dataset" \
--vision_pretrained="sam_vit_h_4b8939.pth" \
--train_datasets="{dataset_name}" \
--val_dataset="{dataset_name}" \
--val_split="test" \
--exp_name="{exp_name}" \
--epochs=20 \
--steps_per_epoch=500 \
--batch_size=4
```

---

## Evaluation

Runs inference on `{dataset_dir}/{val_dataset}/{val_split}.json` and writes results to `{output_dir}/{val_dataset}/`.

```bash
!deepspeed --master_port=24999 train_ds.py \
  --version="xinlai/LISA-13B-llama2-v1" \
  --dataset_dir='./dataset' \
  --vision_pretrained="sam_vit_h_4b8939.pth" \
  --exp_name="lisa-7b" \
  --val_dataset="BUSI" \
  --val_split="test" \
  --resume="./runs/lisa-7b-busi/ckpt_model" \
  --eval_only
```

Output structure:
```
output/BUSI/
├── masks/          ← predicted binary masks (.png)
└── test.json       ← [{image, predicted_mask, answer}, ...]
```

---

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--train_datasets` | `BUSI` | Comma-separated dataset names for training |
| `--val_dataset` | `BUSI` | Dataset for `--eval_only` |
| `--val_split` | `test` | Split used with `--eval_only` |
| `--dataset_dir` | `./dataset` | Root directory of all datasets |
| `--output_dir` | `./output` | Output root for `--eval_only` results |
| `--exp_name` | `medseg` | Run name — sets checkpoint and log path |
| `--vision_pretrained` | — | Path to SAM-family weights (`sam_vit_h_4b8939.pth` or `medsam_vit_b.pth`) |
| `--sam_variant` | `sam_vit_h` | Visual backbone: `sam_vit_h` or `medsam_vit_b` (must match `--vision_pretrained`) |
| `--resume` | — | DeepSpeed checkpoint path to resume from |
| `--epochs` | `10` | Training epochs |
| `--steps_per_epoch` | `500` | Gradient steps per epoch |
| `--batch_size` | `2` | Per-GPU batch size |
| `--grad_accumulation_steps` | `10` | Gradient accumulation |
| `--lr` | `3e-4` | Learning rate |
| `--precision` | `bf16` | `fp32` / `bf16` / `fp16` |
| `--lora_r` | `8` | LoRA rank (0 = disable LoRA) |
| `--eval_only` | `False` | Skip training, run inference only |
| `--no_eval` | `False` | Skip per-epoch validation |
