import sys
sys.path.append('/workspace/wxd/SVD/src')

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from einops import rearrange, repeat
from PIL import Image
import fire
from safetensors import safe_open

from common import json2data

from diffusers.utils import export_to_video

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPModel
from diffuser.models.unet_action_v11 import UNetSpatioTemporalConditionModel_Action
# from diffusers.models.unet_action import UNetSpatioTemporalConditionModel_Action
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffuser.pipelines.stable_video_diffusion.pipeline_action_video_diffusion_v11 import ActionVideoDiffusionPipeline

from transformers import AutoProcessor, AutoModelForCausalLM

def load_models(pretrained_model_name_or_path = '/root/autodl-tmp/smodels-video/sd-temporal-ov-drive-A800/checkpoint-1000/', device='cuda:0'):
    text_encoder = CLIPTextModel.from_pretrained(
                '/volsparse1/wxd/smodels/image-keyframes-ep30', subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
                '/volsparse1/wxd/smodels/image-keyframes-ep30', subfolder="vae",
    )
    # clip_model = CLIPModel.from_pretrained(
    #     "/root/autodl-fs/smodels/clip-vit-large-patch14", torch_dtype=torch.float16)

    text_encoder.eval()
    vae.eval()
    # clip_model.eval()
    text_encoder.to(device)
    vae.to(device)
    # clip_model.to(device)

    tokenizer = CLIPTokenizer.from_pretrained('/volsparse1/wxd/smodels/image-keyframes-ep30', subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained('/volsparse1/wxd/smodels-video/sd-temporal-ov-drive-A100', subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained('/volsparse1/wxd/smodels/image-keyframes-ep30', subfolder="feature_extractor")

    # unet = UNetSpatioTemporalConditionModel_Action.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    # unet.temp_style = "image" # use image clip embedding

    unet = UNetSpatioTemporalConditionModel_Action(cross_attention_dim=768, in_channels=4, temp_style = "action")
    if "unet" not in os.listdir(pretrained_model_name_or_path):
        dm_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors")
    else:
        dm_path = os.path.join(pretrained_model_name_or_path, "unet", "diffusion_pytorch_model.safetensors")
    tensors = {}
    with safe_open(dm_path, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    miss_keys, ignore_keys = unet.load_state_dict(tensors, strict=True)
    print(f'loaded unet from {pretrained_model_name_or_path}')
    print('miss_keys: ', miss_keys)
    print('ignore_keys: ', ignore_keys)

    unet.eval()
    unet = unet.to(device)
    pipeline = ActionVideoDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor
    )
    pipeline = pipeline.to(device).to(torch.float16)
    return pipeline

pretrained_model_name_or_path = '/volsparse1/wxd/smodels-video/sd-temporal-ov-drive-A100/checkpoint-4000'
pipeline = load_models(pretrained_model_name_or_path, "cuda:0")

print('loaded models!\n')

# import pdb; pdb.set_trace()

num_frames = 36
train_frames = 8
device='cuda'
roll_out= 1
width = 448
height = 256


while True:
    prompt = input("prompt: ")
    action = input("action: ")

    image_path = f"/workspace/wxd/SVD/scripts/text_to_video/nusc_test_3.jpg"

    root_dir = "nusc_test_3"
    image = Image.open(image_path)

    video = pipeline(image, num_frames=train_frames, prompt=prompt, prompt_temporal=action, height=height, width=width).frames[0]
    # replace first frame
    # video[0] = image.resize((width, height))
    # for t in range(roll_out):
    #     last_frame = video[-1]
    #     prompt = generate_caption(last_frame, git_processor_large, git_model_large)
    #     video_2 = pipeline(last_frame, num_frames=train_frames, prompt=prompt, action=None, height=height, width=width).frames[0]
    #     video = video + video_2[1:]

    print(f'len of final video {len(video)}')

    name = os.path.basename(image_path).split('.')[0] + action.replace(" ", "_").strip()

    os.makedirs(os.path.join(root_dir), exist_ok=True)

    export_to_video(video, os.path.join(root_dir, f'{name}.mp4'), fps=6)
