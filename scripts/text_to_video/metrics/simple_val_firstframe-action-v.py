# NOTE
# This file is for aliyun 1-GPU test

import sys
sys.path.append('/mnt/cache/wangxiaodong/SDM/src')

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
import json

from common import json2data

from diffusers.utils import export_to_video

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.models.unet_action_v14 import UNetSpatioTemporalConditionModel_Action
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_video_diffusion.pipeline_action_video_diffusion_v1_v import ActionVideoDiffusionPipeline
# fix fps=3

from transformers import AutoProcessor, AutoModelForCausalLM

def load_models(pretrained_model_name_or_path = '/mnt/lustrenew/wangxiaodong/smodels-vis/video-v01-v-ep100-s192-5e-5', device='cuda:0'):
    text_encoder = CLIPTextModel.from_pretrained(
                '/mnt/lustrenew/wangxiaodong/smodels/image-ep50-ddp', subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
                '/mnt/lustrenew/wangxiaodong/smodels/image-ep50-ddp', subfolder="vae"
    )

    text_encoder.eval()
    vae.eval()
    text_encoder.to(device)
    vae.to(device)

    tokenizer = CLIPTokenizer.from_pretrained('/mnt/lustrenew/wangxiaodong/smodels/image-ep50-ddp', subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained('/mnt/lustrenew/wangxiaodong/smodels/image-ep50-ddp', subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained('/mnt/lustrenew/wangxiaodong/smodels/image-ep50-ddp', subfolder="feature_extractor")

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
    pipeline = pipeline.to(device).to(torch.float16)
    return pipeline

def generate_caption(image, git_processor_large, git_model_large, device='cuda:0'):

    git_model_large.to(device)
    pil_list = [image.convert('RGB')]
    inputs = git_processor_large(images=pil_list, return_tensors="pt").to(device)
    
    generated_ids = git_model_large.generate(pixel_values=inputs.pixel_values, max_length=50)

    generated_caption = git_processor_large.batch_decode(generated_ids, skip_special_tokens=True)
   
    return generated_caption[0]

def main(
    pretrained_model_name_or_path = '/mnt/lustrenew/wangxiaodong/smodels-vis/video-v01-v-ep100-s192-5e-5',
    num_frames = 15,
    root_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/FVD-first',
    train_frames = 8,
    device='cuda',
    roll_out= 5
):
    pipeline = load_models(pretrained_model_name_or_path, device)

    root_dir = root_dir + f'-{num_frames}-rec'

    os.makedirs(root_dir, exist_ok=True)

    print('loaded pipeline!')
    
    meta_data = json2data('/mnt/lustrenew/wangxiaodong/data/nuscene/samples_group_sort_val.json')
    files_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/val_group'

    json_path = '/mnt/lustrenew/wangxiaodong/data/nuscene/frame_action_val.json'
    with open(json_path, 'r') as f:
        frame_action = json.load(f)

    version = os.path.basename(pretrained_model_name_or_path) + '-gt'

    os.makedirs(os.path.join(root_dir, version), exist_ok=True)

    # caption model
    git_processor_large = AutoProcessor.from_pretrained("/mnt/lustrenew/wangxiaodong/models/git-large-coco")
    git_model_large = AutoModelForCausalLM.from_pretrained("/mnt/lustrenew/wangxiaodong/models/git-large-coco")
    print('loaded caption model!')

    for item in tqdm(meta_data):
        sce = item['scene']
        samples = item['samples']
        file = samples[0]
        if file.split('.')[-1] == 'jpg' or file.split('.')[-1] == 'png':
            pass
        else:
            continue
        image_path = os.path.join(files_dir, sce, file)
        image = Image.open(image_path)
        prompt = generate_caption(image, git_processor_large, git_model_large)

        scene_data = frame_action[sce]
        steerss = scene_data[0]
        speedss = scene_data[1]

        seek_steer = steerss[0: 8]
        seek_speed = speedss[0: 8]
        seek_steer = [0.] + seek_steer[1:]
        seek_speed = [0.] + seek_speed[1:]

        seek_steer = torch.tensor(seek_steer)
        seek_speed = torch.tensor(seek_speed)

        steer_scale = 10.
        steer_offset = 90.
        speed_scale = 10.
        speed_offset = 2.

        steers = seek_steer.to(device) * steer_scale + steer_offset
        speeds = seek_speed.to(device) * speed_scale + speed_offset
        steers = steers.view(-1, 1)
        speeds = speeds.view(-1, 1)
        action = torch.cat([steers, speeds], dim=-1)

        action = action.unsqueeze(0)

        video = pipeline(image, num_frames=train_frames, prompt=prompt, action=action, height=192, width=384).frames[0]
        # use gt frame
        # 8 + 8*4
        gt_idx = 0
        for t in range(roll_out):
            gt_idx = gt_idx + 7

            seek_steer = steerss[gt_idx: gt_idx+8]
            seek_speed = speedss[gt_idx: gt_idx+8]
            seek_steer = [0.] + seek_steer[1:]
            seek_speed = [0.] + seek_speed[1:]

            seek_steer = torch.tensor(seek_steer)
            seek_speed = torch.tensor(seek_speed)

            steers = seek_steer.to(device) * steer_scale + steer_offset
            speeds = seek_speed.to(device) * speed_scale + speed_offset
            steers = steers.view(-1, 1)
            speeds = speeds.view(-1, 1)
            action = torch.cat([steers, speeds], dim=-1)

            action = action.unsqueeze(0)

            # tmp_path = os.path.join(files_dir, sce, samples[gt_idx])
            last_frame = video[-1]
            prompt = generate_caption(last_frame, git_processor_large, git_model_large)
            video_2 = pipeline(last_frame, num_frames=train_frames, prompt=prompt, action=action, height=192, width=384).frames[0]
            video = video + video_2[1:]
        
        print(f'len of final video {len(video)}')

        name = os.path.basename(image_path).split('.')[0]

        os.makedirs(os.path.join(root_dir, version, sce), exist_ok=True)

        export_to_video(video, os.path.join(root_dir, version, sce, f'{name}.mp4'), fps=6)
        # print(f'save to duplicate name')

    print('inference done!')

if __name__ == '__main__':
    fire.Fire(main)