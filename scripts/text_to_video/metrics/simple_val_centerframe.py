# NOTE
# This file is for aliyun 1-GPU test

import sys
sys.path.append('/mnt/workspace/diffusers/src')

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

from diffusers.utils import export_to_video

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.models.unet_action import UNetSpatioTemporalConditionModel_Action
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_video_diffusion.pipeline_action_video_diffusion_v1 import ActionVideoDiffusionPipeline

from transformers import AutoProcessor, AutoModelForCausalLM

def load_models(pretrained_model_name_or_path = '/mnt/workspace/drive-video-s256-v01-ep100', device='cuda:0'):
    text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path, subfolder="vae"
    )

    text_encoder.eval()
    vae.eval()
    text_encoder.to(device)
    vae.to(device)

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor")

    unet = UNetSpatioTemporalConditionModel_Action.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
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
    pipeline = pipeline.to(device)
    return pipeline

def generate_caption(image, git_processor_large, git_model_large, device='cuda:0'):

    git_model_large.to(device)
    pil_list = [image.convert('RGB')]
    inputs = git_processor_large(images=pil_list, return_tensors="pt").to(device)
    
    generated_ids = git_model_large.generate(pixel_values=inputs.pixel_values, max_length=50)

    generated_caption = git_processor_large.batch_decode(generated_ids, skip_special_tokens=True)
   
    return generated_caption[0]

def main(
    pretrained_model_name_or_path = '/mnt/workspace/drive-video-s256-v01-ep100',
    num_frames = 8,
    root_dir = '/mnt/workspace/FVD-center',
):
    pipeline = load_models(pretrained_model_name_or_path)

    root_dir = root_dir + f'-{num_frames}'

    os.makedirs(root_dir, exist_ok=True)

    print('loaded pipeline!')
    files_dir = '/mnt/workspace/val_150_centerframes'
    files = os.listdir(files_dir)

    version = os.path.basename(pretrained_model_name_or_path)

    os.makedirs(os.path.join(root_dir, version), exist_ok=True)

    # caption model
    git_processor_large = AutoProcessor.from_pretrained("/mnt/workspace/git-large-coco")
    git_model_large = AutoModelForCausalLM.from_pretrained("/mnt/workspace/git-large-coco")
    print('loaded caption model!')

    for file in tqdm(files):
        if file.split('.')[-1] == 'jpg' or file.split('.')[-1] == 'png':
            pass
        else:
            continue
        image_path = os.path.join(files_dir, file)
        image = Image.open(image_path)
        prompt = generate_caption(image, git_processor_large, git_model_large)

        video = pipeline(image, num_frames=num_frames, prompt=prompt, action=None, height=256, width=512).frames[0]

        name = os.path.basename(image_path).split('.')[0]

        export_to_video(video, os.path.join(root_dir, version, f'{name}.mp4'), fps=6)

    print('inference done!')

if __name__ == '__main__':
    fire.Fire(main)