#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import json
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import sys
sys.path.append('/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/src')


import diffuser
from diffuser import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline

from llama.attention_custom import LlamaModeling

from diffuser.optimization import get_scheduler
from diffuser.training_utils import EMAModel, compute_snr
from diffuser.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffuser.utils.import_utils import is_xformers_available


from nuscene_action_llama_ms import Actionframes

from safetensors import safe_open
from collections import OrderedDict

from einops import rearrange, repeat

if is_wandb_available():
    import wandb


def numpy_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    raise TypeError(f"{type(obj)} is not JSON serializable")



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def save_model_card(
    args,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels/image-ep50-s192-1e-4",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_clip_model_name_or_path",
        type=str,
        default="/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels/clip-vit-large-patch14",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="/ssd_datasets/wxiaodong/naruto-blip-captions",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="action-vae",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=14, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=20, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default='v_prediction',
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="Llama_action_train",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--drop_context",
        type=float,
        default=0.1,
        help="ratio for drop image context"
    )
    parser.add_argument(
        "--temp_style",
        type=str,
        default="image",
        help=(
            "temp layer cross-attention information, text, text-image, image"
        ),
    )
    parser.add_argument(
        "--img_style",
        type=str,
        default="pooler_output",
        help=(
            "image clip embedding type, pooler_output, last_hidden_state"
        ),
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=8,
        help="history context"
    )
    parser.add_argument(
        "--ac_beta",
        type=float,
        default=0.01,
        help="beta for action loss"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="resume checkpoint path"
    )
    parser.add_argument(
        "--stop_action_grad",
        action="store_true",
        help="Whether or not to stop action branch gradient",
    )
    parser.add_argument(
        "--loss",
        default="l1",
        help="l1 loss, l2 loss",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help=(
            "eval training model frequency"
        ),
    )
    parser.add_argument(
        "--skip_image",
        action="store_true",
        help="Whether or not to skip image input as guidance",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=3,
        help="action context context"
    )



    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def main():
    args = parse_args()

    args.output_dir = os.path.join('/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels-vis', args.output_dir)

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffuser.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffuser.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # text_encoder = CLIPTextModel.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        # )
        if not args.skip_image:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
            )


    # for action
    from action_llama_config import LlamaConfig
    config = LlamaConfig()
    action_model = LlamaModeling(config)
    action_model.train()

    if args.ckpt:
        # load checkpoint
        action_model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        if accelerator.is_main_process:
            print(f'loaded weights from {args.ckpt}')

    # update action_model
    optimize_param = []
    for name, param in action_model.named_parameters():
        param.requires_grad = True
        optimize_param.append(param)

    if not args.skip_image:
        # Freeze vae and text_encoder and set unet to trainable
        vae.requires_grad_(False)
        vae.eval()

    params = sum(p.numel() for p in action_model.parameters() if p.requires_grad)
    params_in_million = params / 1e6  # M
    if accelerator.is_main_process:
        print(f"updated parameters: {params_in_million} M")


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            action_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        action_model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        optimize_param,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    with accelerator.main_process_first():
        train_dataset = Actionframes(split='train', args=config, tokenizer=tokenizer, img_size=(512, 512), history_len = config.history_len, max_video_len = config.max_video_len)
        val_dataset = Actionframes(split='val', args=config, tokenizer=tokenizer, img_size=(512, 512), max_video_len = config.max_video_len)
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=8,
        drop_last=False
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    action_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        action_model, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    
    if not args.skip_image:
        # Move text_encode and vae to gpu and cast to weight_dtype
        vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Learning Rate = {args.learning_rate}")
    global_step = 0
    first_epoch = 0

    
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(action_model):
                if not args.skip_image:
                    video_frames = batch["label_imgs"].to(weight_dtype)
                    bs, l, _, _, _ = video_frames.size()
                    video_frames = rearrange(video_frames, 'b l c h w -> (b l) c h w', b=bs, l=l)

                    latents = vae.encode(video_frames).latent_dist.sample() # receive bl, c, h, w
                    latents = latents * vae.config.scaling_factor

                    latents = rearrange(latents, '(b l) c h w -> b l c h w', b=bs, l=l)

                    # image context
                    image_context = latents[:, 0] # get first frame latents # (b, 4, h, w)

                    del latents
                else:
                    image_context = None

                # NOTE For Action
                steers = batch['steer'] # b, f
                speeds = batch['speed'] # b, f
                attention_mask = batch['attention_mask']
                loss_mask = batch['loss_mask']


                action = torch.stack([steers, speeds], dim=-1) # b, f, 2

                src_action = action[:, :-1] # b f-1 1

                # Predict the noise residual and compute loss
                hidden_states = action_model(src_action, attention_mask, image_context)
                logits = hidden_states['logits']
                logits = logits.float()

                labels = action[..., :, :].contiguous()

                # shift_logits = logits[..., :, :].contiguous()
                # shift_labels = labels[..., :].contiguous()
                # loss_mask = loss_mask[..., :].contiguous()
                shift_logits = logits
                shift_labels = labels
                loss_mask = loss_mask

                bsz = loss_mask.shape[0]
                
                
                if args.loss == 'l2':
                    loss_action = F.mse_loss(shift_logits, shift_labels, reduction="none")
                    loss_action = loss_action.mean(dim=-1)
                    loss_action = (loss_action.view(bsz, -1) * loss_mask).mean()
                elif args.loss == 'l1':
                    loss_action = F.l1_loss(shift_logits, shift_labels, reduction="none")
                    loss_action = loss_action.mean(dim=-1)
                    loss_action = (loss_action.view(bsz, -1) * loss_mask).mean()
                else:
                    raise ValueError(f"{args.loss} is not implemented")
                loss = loss_action

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(action_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # NOTE validation
                if global_step % args.eval_steps == 0:
                    if accelerator.is_main_process:
                        action_model.eval()
                        device = next(action_model.parameters()).device
                        
                        gather_preds = []
                        gather_gt = []
                        gather_his = []
                        
                        action_compare = []

                        n = 0
                        for n, batch in tqdm(enumerate(val_dataloader)):
                            n = n + 1
                            # if n>20:
                            #     break
                            names = batch['name']
                            scenes = batch['scene']
                            with torch.no_grad():
                                if not args.skip_image:
                                    video_frames = batch["label_imgs"].to(weight_dtype)
                                    bs, l, _, _, _ = video_frames.size()
                                    video_frames = rearrange(video_frames, 'b l c h w -> (b l) c h w', b=bs, l=l)

                                    video_frames = video_frames.to(device)
                                    latents = vae.encode(video_frames).latent_dist.sample() # receive bl, c, h, w
                                    latents = latents * vae.config.scaling_factor

                                    latents = rearrange(latents, '(b l) c h w -> b l c h w', b=bs, l=l)
                                    del latents
                                    # image context
                                    image_context = latents[:, 0] # get first frame latents # (b, 4, h, w)
                                else:
                                    image_context = None
                                
                                steers = batch['steer'] # b, f
                                speeds = batch['speed'] # b, f

                                action = torch.stack([steers, speeds], dim=-1) # b, f, 2

                                history_action = action[:, ]
                                future_action = action[:, config.history_len:]

                                future_action = future_action.to(device)

                                gather_gt.append(future_action)
                                gather_his.append(history_action)

                                history_action = history_action.to(device)

                                pred_action = []
                                # TODO
                                # test ctx3 -> 1 prediction
                                # compare with ADrive-I
                                # [0,1,2->3, 1,2,3->4]
                                prediction_scope = 4 # NOTE use GT to predict how long
                                print(f'take {config.history_len} history action, predict scope {prediction_scope}')
                                for ac_index in tqdm(range(0, len(action[0]-config.history_len), prediction_scope)):
                                    
                                    # for each timestep
                                    # sub_history_action = history_action[:, ac_index-config.history_len: ac_index]
                                    # action gap, 0,1,2 -> pred (3,4,5) GT -> pred (6,7,8)
                                    sub_history_action = history_action[:, ac_index: ac_index+config.history_len]

                                    history_action_ = sub_history_action

                                    # prepare attention_mask
                                    input_mask = [1] + [1] * sub_history_action.shape[1] # bos + history_action
                                    input_mask = torch.Tensor(input_mask).to(device).unsqueeze(0)
                                    
                                    prompt_len = len(input_mask[0])
                                    start_pos = prompt_len
                                    total_len = prompt_len + prediction_scope # NOTE hard to set 1

                                    attention_mask = input_mask.bool()
                                    attention_mask = torch.cat(
                                        (attention_mask, torch.zeros(1, total_len - attention_mask.shape[-1]).bool().to(attention_mask.device)),
                                        dim=-1)

                                    prev_pos = 0
                                    past_key_values = None
                                    input_embed = action_model.embed_tokens(history_action_)

                                    # add bos
                                    input_embed = torch.cat([action_model.bos_embed, input_embed], dim=1) 

                                    next_ipt = input_embed[:, prev_pos:start_pos]

                                    for cur_pos in range(start_pos, total_len):
                                        intermid_state = action_model(
                                            inputs_embeds=next_ipt,
                                            past_key_values=past_key_values,
                                            use_cache=True,
                                        )
                                        inter_hidden_states = intermid_state["hidden_states"][:, -1, :]  # only keep the last token
                                        prediction = action_model.lm_head(inter_hidden_states) # b 1 2
                                        past_key_values = intermid_state["next_cache"]

                                        pred_action.append(prediction[:, None])

                                        prev_pos = cur_pos
                                        next_ipt = action_model.embed_tokens(prediction)[:, None, :]
                                    # each timestep inference
                                    
                                pred_action = torch.cat(pred_action, dim=1) # b f d
                                
                                gather_preds.append(pred_action)

                                for idx_n, name in enumerate(names):
                                    action_hat_sam = pred_action[idx_n]
                                    action_gt_sam = future_action[idx_n]
                                    action_his_sam = history_action[idx_n]
                                    action_compare.append(
                                        {
                                            'scene': scenes[idx_n],
                                            'name': name,
                                            'action_pred': action_hat_sam.cpu().numpy(),
                                            'action_gt': action_gt_sam.cpu().numpy(),
                                            'action_his': action_his_sam.cpu().numpy()
                                        }
                                    )

                        # gather
                        gather_preds = torch.cat(gather_preds, dim=0) # tbsz, f, d
                        gather_gt = torch.cat(gather_gt, dim=0) # tbsz, f, d
                        print(f'pred shape: {gather_preds.shape}')
                        print(f'gt shape: {gather_gt.shape}')
                        gather_preds = gather_preds[:, :gather_gt.shape[1]] # cut off
                        discrepancy = F.l1_loss(gather_preds.flatten(0, 1).float(), gather_gt.flatten(0, 1).float(), reduction="mean")
                        t0_discrepancy = F.l1_loss(gather_preds[:, 0].float(), gather_gt[:, 0].float(), reduction="mean")

                        print(f'--------------------------')
                        print(f'eval l1 discrepancy: {discrepancy}, t0 discrepancy: {t0_discrepancy}')

                        accelerator.log({"val_loss": discrepancy}, step=global_step)
                        accelerator.log({"val_t0_loss": t0_discrepancy}, step=global_step)

                        print(f'take {config.history_len} history action, predict scope {prediction_scope}')

                        os.makedirs(os.path.join(args.output_dir, f'preview_{global_step}step'), exist_ok=True)

                        # save prediction
                        with open(os.path.join(args.output_dir, f'preview_{global_step}step', 'action.json'), 'w') as f:
                            json.dump(action_compare, f, default=numpy_serializable)

                        print('inference done!')


                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        file_path = os.path.join(save_path, 'ckpt.pth')
                        unwrapped_unet = accelerator.unwrap_model(action_model)
                        torch.save(unwrapped_unet.state_dict(), file_path)
                        logger.info(f"Saved state to {file_path}")
                        del unwrapped_unet

                action_model.train()
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(action_model)

        #TODO save UNet
        # save for ZeRO-3 offloading
        # unwrapped_unet.save_pretrained(
        #     os.path.join(args.output_dir, 'unet'),
        #     is_main_process=accelerator.is_main_process,
        #     save_function=accelerator.save,
        #     state_dict=accelerator.get_state_dict(unet),
        # )
        file_path = os.path.join(args.output_dir, 'last.pth')
        torch.save(unwrapped_unet.state_dict(), file_path)
        logger.info(f"Saved state to {file_path}")
        del unwrapped_unet

        # from safetensors.torch import save_file
        #TODO ZeRO-2 saving
        # unwrapped_unet.save_pretrained(os.path.join(args.output_dir, 'action_vae'))


    accelerator.end_training()


if __name__ == "__main__":
    main()
