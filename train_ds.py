import argparse
import json
import os
import shutil
import sys
import time
from functools import partial

os.environ["MPLBACKEND"] = "agg"
import matplotlib
matplotlib.use("agg")

import cv2
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.dataset import MedDataset, collate_fn
from utils.text_metrics import compute_all as compute_text_metrics
from utils.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    AverageMeter, ProgressMeter, Summary, dict_to_cuda,
    intersectionAndUnionGPU,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Medical Segmentation Training")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # --- dataset ---
    parser.add_argument(
        "--train_datasets", default="BUSI", type=str,
        help="Comma-separated dataset names for training. "
             "Loads {dataset_dir}/{name}/train.json for each.",
    )
    parser.add_argument(
        "--val_dataset", default="BUSI", type=str,
        help="Dataset name for --eval_only. "
             "Loads {dataset_dir}/{name}/{val_split}.json and saves output.",
    )
    parser.add_argument("--val_split", default="test", choices=["val", "test"],
                        help="Split used with --eval_only (default: test).")
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="Root dir for eval_only results: {output_dir}/{dataset_name}/")

    # --- training ---
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="medseg", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--sam_variant", default="sam_vit_h", type=str,
                        choices=["sam_vit_h", "medsam_vit_b"],
                        help="Which SAM-family encoder to use as the visual backbone.")
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_val_loader(ds_name, split, args, tokenizer, _collate):
    """Build a DataLoader for one validation dataset. Returns None if JSON missing."""
    json_path = os.path.join(args.dataset_dir, ds_name, f"{split}.json")
    if not os.path.exists(json_path):
        print(f"[warn] {json_path} not found — skipping {ds_name}")
        return None
    ds = MedDataset(
        json_paths=[json_path],
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        base_dir=args.dataset_dir,
        image_size=args.image_size,
        inference=True,
        sam_variant=args.sam_variant,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False, drop_last=False)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=sampler,
        collate_fn=_collate,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.vis_save_path, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # ---- distributed process group (needed by DistributedSampler before deepspeed.initialize) ----
    deepspeed.init_distributed()

    # ---- tokenizer ----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    # ---- model ----
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "sam_variant": args.sam_variant,
    }
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.half}.get(args.precision, torch.float32)
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ---- LoRA ----
    if args.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, torch.nn.Linear)
                    and all(x not in name for x in ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"])
                    and any(x in name for x in lora_target_modules)
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_linear_layers(model, args.lora_target_modules.split(",")),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]):
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    _collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        conv_type=args.conv_type,
        use_mm_start_end=args.use_mm_start_end,
        local_rank=args.local_rank,
    )

    # ---- training dataset ----
    if not args.eval_only:
        train_json_paths = [
            os.path.join(args.dataset_dir, name.strip(), "train.json")
            for name in args.train_datasets.split(",")
        ]
        train_dataset = MedDataset(
            json_paths=train_json_paths,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            base_dir=args.dataset_dir,
            image_size=args.image_size,
            inference=False,
            sam_variant=args.sam_variant,
        )
        print(f"Training: {len(train_dataset)} samples")
    else:
        train_dataset = None

    # ---- val loaders for per-epoch evaluation (only the training datasets) ----
    val_loaders = {}   # {dataset_name: DataLoader}
    if not args.no_eval and not args.eval_only:
        for ds_name in [n.strip() for n in args.train_datasets.split(",")]:
            loader = _make_val_loader(ds_name, "val", args, tokenizer, _collate)
            if loader is not None:
                val_loaders[ds_name] = loader
                print(f"Val [{ds_name}]: {len(loader.dataset)} samples")

    # ---- eval_only: single dataset, saves output JSON ----
    if args.eval_only:
        eval_loader = _make_val_loader(args.val_dataset, args.val_split, args, tokenizer, _collate)

    # ---- DeepSpeed ----
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "weight_decay": 0.0, "betas": (args.beta1, args.beta2)},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    ds_init_kwargs = dict(model=model, model_parameters=model.parameters(), config=ds_config)
    if not args.eval_only:
        ds_init_kwargs["training_data"] = train_dataset
        ds_init_kwargs["collate_fn"] = _collate
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(**ds_init_kwargs)

    # ---- resume ----
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume
    if args.resume:
        model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest")) as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resumed from {args.resume}, starting at epoch {args.start_epoch}")

    # ---- eval only ----
    if args.eval_only:
        validate(eval_loader, model_engine, 0, writer, args, tokenizer,
                 dataset_name=args.val_dataset, save_output=True)
        return

    # ---- training loop ----
    best_score, cur_ciou = 0.0, 0.0
    train_iter = iter(train_loader)

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(train_loader, model_engine, epoch, scheduler, writer, train_iter, args)

        if val_loaders:
            scores = []
            for ds_name, v_loader in val_loaders.items():
                giou, ciou = validate(
                    v_loader, model_engine, epoch, writer, args, tokenizer,
                    dataset_name=ds_name,
                )
                scores.append(giou)
            avg_giou = sum(scores) / len(scores)
            is_best = avg_giou > best_score
            best_score = max(avg_giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou  # ciou from last dataset
        else:
            is_best = True  # always save when no eval

        if is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)

    # ---- post-training test on {val_dataset}/{val_split} ----
    if not args.eval_only:
        test_loader = _make_val_loader(args.val_dataset, args.val_split, args, tokenizer, _collate)
        if test_loader is not None:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if os.path.exists(save_dir):
                model_engine.load_checkpoint(save_dir)
                if args.local_rank == 0:
                    print(f"Loaded best checkpoint from {save_dir} for test")
            validate(test_loader, model_engine, args.epochs, writer, args, tokenizer,
                     dataset_name=args.val_dataset, save_output=True)
            if args.distributed:
                torch.distributed.barrier()
            


# train

def train(train_loader, model, epoch, scheduler, writer, train_iter, args):
    batch_time       = AverageMeter("Time",        ":6.3f")
    data_time        = AverageMeter("Data",        ":6.3f")
    losses           = AverageMeter("Loss",        ":.4f")
    ce_losses        = AverageMeter("CeLoss",      ":.4f")
    mask_bce_losses  = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss",":.4f")
    mask_losses      = AverageMeter("MaskLoss",    ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, ce_losses, mask_losses, mask_bce_losses, mask_dice_losses],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss           = output_dict["loss"]
            ce_loss        = output_dict["ce_loss"]
            mask_bce_loss  = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss      = output_dict["mask_loss"]

            n = input_dict["images"].size(0)
            losses.update(loss.item(), n)
            ce_losses.update(ce_loss.item(), n)
            mask_bce_losses.update(mask_bce_loss.item(), n)
            mask_dice_losses.update(mask_dice_loss.item(), n)
            mask_losses.update(mask_loss.item(), n)
            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                for m in [batch_time, data_time, losses, ce_losses,
                          mask_bce_losses, mask_dice_losses, mask_losses]:
                    m.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss",            losses.avg,           global_step)
                writer.add_scalar("train/ce_loss",         ce_losses.avg,        global_step)
                writer.add_scalar("train/mask_bce_loss",   mask_bce_losses.avg,  global_step)
                writer.add_scalar("train/mask_dice_loss",  mask_dice_losses.avg, global_step)
                writer.add_scalar("train/mask_loss",       mask_losses.avg,      global_step)
                writer.add_scalar("metrics/secs_per_batch", batch_time.avg,      global_step)
                writer.add_scalar("metrics/data_secs",      data_time.avg,       global_step)

            for m in [batch_time, data_time, losses, ce_losses,
                      mask_bce_losses, mask_dice_losses, mask_losses]:
                m.reset()

        if global_step != 0 and args.local_rank == 0:
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    return train_iter



# validate


def _build_gen_prompt(question: str, args) -> str:
    """Build a single-turn LISA prompt with an empty assistant slot for generation."""
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question
    if args.use_mm_start_end:
        user_msg = user_msg.replace(
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
        )
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], "")
    return conv.get_prompt()


def _clean_generated_answer(text: str) -> str:
    """Strip [SEG], control whitespace, and trailing punctuation from a generated answer."""
    # Truncate if the model rolled into a second turn.
    for stop in ("USER:", "ASSISTANT:", "<|im_end|>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    text = text.replace("[SEG]", "")
    text = text.replace("</s>", "")
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip(" .")


def validate(val_loader, model_engine, epoch, writer, args, tokenizer,
             dataset_name="val", save_output=False):
    """
    Run one validation pass.

    dataset_name  : used for TensorBoard keys (val/{dataset_name}/...) and
                    the output directory when save_output=True.
    save_output   : when True (eval_only mode) saves predicted masks to
                      {output_dir}/{dataset_name}/masks/<stem>_pred.png
                    and writes a JSON summary to
                      {output_dir}/{dataset_name}/test.json
                    Each entry: {"image", "question", "predicted_mask",
                                 "answer", "predicted_answer",
                                 "bleu1", "rouge_l", "f1"}
    """
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    acc_dice_meter = AverageMeter("gDice", ":6.3f", Summary.SUM)
    bleu_meter  = AverageMeter("BLEU-1",  ":6.4f", Summary.AVERAGE)
    rouge_meter = AverageMeter("ROUGE-L", ":6.4f", Summary.AVERAGE)
    f1_meter    = AverageMeter("F1",      ":6.4f", Summary.AVERAGE)

    model_engine.eval()

    results = []   # populated only when save_output=True
    mask_out_dir = os.path.join(args.output_dir, dataset_name, "masks")
    if save_output and args.local_rank == 0:
        os.makedirs(mask_out_dir, exist_ok=True)

    for input_dict in tqdm.tqdm(val_loader, desc=dataset_name):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks  = output_dict["pred_masks"]
        masks_list  = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        # ---- QA: generate answer text from the question alone ----
        raw_question = input_dict["raw_questions_list"][0]
        gt_answer    = input_dict["questions_list"][0]

        gen_prompt = _build_gen_prompt(raw_question, args)
        gen_input_ids = tokenizer_image_token(
            gen_prompt, tokenizer, return_tensors="pt"
        ).unsqueeze(0).cuda()

        gt_mask_hw = input_dict["masks_list"][0].shape[-2:]
        original_size_list = [(int(gt_mask_hw[0]), int(gt_mask_hw[1]))]

        with torch.no_grad():
            gen_output_ids, _ = model_engine.module.evaluate(
                input_dict["images_clip"],
                input_dict["images"],
                gen_input_ids,
                input_dict["resize_list"],
                original_size_list,
                max_new_tokens=128,
                tokenizer=tokenizer,
            )

        # generated tokens follow the prompt; slice them off and decode
        new_ids = gen_output_ids[0][gen_input_ids.shape[1]:]
        pred_answer = tokenizer.decode(new_ids, skip_special_tokens=True)
        pred_answer = _clean_generated_answer(pred_answer)

        text_m = compute_text_metrics(pred_answer, gt_answer)
        bleu_meter.update(text_m["bleu1"])
        rouge_meter.update(text_m["rouge_l"])
        f1_meter.update(text_m["f1"])

        #  save output (eval_only, rank 0)
        if save_output and args.local_rank == 0:
            image_path = input_dict["image_paths"][0]
            image_stem = os.path.splitext(os.path.basename(image_path))[0]

            # save predicted binary mask (resize back to original image resolution)
            pred_np = (pred_masks[0][0].detach().float() > 0).cpu().numpy().astype(np.uint8)
            orig = cv2.imread(image_path)
            if orig is not None and pred_np.shape != orig.shape[:2]:
                pred_np = cv2.resize(
                    pred_np, (orig.shape[1], orig.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_rel  = os.path.join(dataset_name, "masks", f"{image_stem}_pred.png")
            mask_abs  = os.path.join(args.output_dir, mask_rel)
            cv2.imwrite(mask_abs, pred_np * 255)

            results.append({
                "image":            image_path,
                "question":         raw_question,
                "predicted_mask":   mask_rel,
                "answer":           gt_answer,
                "predicted_answer": pred_answer,
                "bleu1":            text_m["bleu1"],
                "rouge_l":          text_m["rouge_l"],
                "f1":               text_m["f1"],
            })

        # metrics
        intersection, union, acc_iou, acc_dice = 0.0, 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union        += union_i

            acc_iou_i = intersection_i / (union_i + 1e-5)
            acc_iou_i[union_i == 0] = 1.0
            acc_iou += acc_iou_i

            dice_i = 2 * intersection_i / (union_i + intersection_i + 1e-5)
            dice_i[union_i == 0] = 1.0
            acc_dice += dice_i

        n = masks_list.shape[0]
        acc_dice_meter.update(acc_dice.cpu().numpy() / n, n=n)
        intersection_meter.update(intersection.cpu().numpy())
        union_meter.update(union.cpu().numpy())
        acc_iou_meter.update(acc_iou.cpu().numpy() / n, n=n)

    # across GPUs (since using deepspeed)
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    acc_dice_meter.all_reduce()
    bleu_meter.all_reduce()
    rouge_meter.all_reduce()
    f1_meter.all_reduce()

    iou_class  = intersection_meter.sum / (union_meter.sum + 1e-10)
    dice_class = 2 * intersection_meter.sum / (union_meter.sum + intersection_meter.sum + 1e-10)
    ciou  = iou_class[1]
    giou  = acc_iou_meter.avg[1]
    cdice = dice_class[1]
    gdice = acc_dice_meter.avg[1]
    bleu1  = bleu_meter.avg
    rougel = rouge_meter.avg
    f1     = f1_meter.avg

    if args.local_rank == 0:
        if writer is not None:
            writer.add_scalar(f"val/{dataset_name}/giou",    giou,   epoch)
            writer.add_scalar(f"val/{dataset_name}/ciou",    ciou,   epoch)
            writer.add_scalar(f"val/{dataset_name}/gdice",   gdice,  epoch)
            writer.add_scalar(f"val/{dataset_name}/cdice",   cdice,  epoch)
            writer.add_scalar(f"val/{dataset_name}/bleu1",   bleu1,  epoch)
            writer.add_scalar(f"val/{dataset_name}/rouge_l", rougel, epoch)
            writer.add_scalar(f"val/{dataset_name}/f1",      f1,     epoch)
        print(f"[{dataset_name}] giou: {giou:.4f}  ciou: {ciou:.4f}  "
              f"gdice: {gdice:.4f}  cdice: {cdice:.4f}  "
              f"bleu1: {bleu1:.4f}  rouge_l: {rougel:.4f}  f1: {f1:.4f}")

        # json output (eval_only)
        if save_output and results:
            out_json = os.path.join(args.output_dir, dataset_name, "test.json")
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} results → {out_json}")

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
