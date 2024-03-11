# NOTE
# This file is for aliyun 1-GPU test

import sys
sys.path.append('/mnt/cache/wangxiaodong/SDM/src')
sys.path.append('/mnt/cache/wangxiaodong/SDM')

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

from diffusers.utils import export_to_video

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPModel
from diffusers.models.unet_action_v3_0 import UNetSpatioTemporalConditionModel_Action
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_action_to_video_diffusion_v2 import ActionVideoDiffusionPipeline

from scripts.action_to_video.nuscene_action_val import Actionframes

from transformers import AutoProcessor, AutoModelForCausalLM

def load_models(pretrained_model_name_or_path = '/mnt/lustrenew/wangxiaodong/smodels-vis/action-v30--5e-5/checkpoint-6000', device='cuda:0'):
    text_encoder = CLIPTextModel.from_pretrained(
                '/mnt/lustrenew/wangxiaodong/smodels/image-ep50-s192-1e-4', subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
                '/mnt/lustrenew/wangxiaodong/smodels/image-ep50-s192-1e-4', subfolder="vae"
    )
    clip_model = CLIPModel.from_pretrained(
        "/mnt/lustrenew/wangxiaodong/models/clip-vit-large-patch14", torch_dtype=torch.float16)

    text_encoder.eval()
    vae.eval()
    clip_model.eval()
    text_encoder.to(device)
    vae.to(device)
    clip_model.to(device)

    tokenizer = CLIPTokenizer.from_pretrained('/mnt/lustrenew/wangxiaodong/smodels/image-ep50-s192-1e-4', subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained('/mnt/lustrenew/wangxiaodong/smodels/image-ep50-s192-1e-4', subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained('/mnt/lustrenew/wangxiaodong/smodels/image-ep50-s192-1e-4', subfolder="feature_extractor")

    unet = UNetSpatioTemporalConditionModel_Action(cross_attention_dim=768, in_channels=4, temp_style = "image")
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
            feature_extractor=feature_extractor,
            clip_model=clip_model
    )
    pipeline = pipeline.to(device).to(torch.float16)
    return pipeline, tokenizer

def generate_caption(image, git_processor_large, git_model_large, device='cuda:0'):

    git_model_large.to(device)
    pil_list = [image.convert('RGB')]
    inputs = git_processor_large(images=pil_list, return_tensors="pt").to(device)
    
    generated_ids = git_model_large.generate(pixel_values=inputs.pixel_values, max_length=50)

    generated_caption = git_processor_large.batch_decode(generated_ids, skip_special_tokens=True)
   
    return generated_caption[0]

def main(
    pretrained_model_name_or_path = '/mnt/lustrenew/wangxiaodong/smodels-vis/action-v30--5e-5/checkpoint-6000',
    num_frames = 8,
    root_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/FVD-first',
    train_frames = 8,
    device='cuda',
    roll_out= 1,
    cfg_min = 1.0,
    cfg_max = 1.0,
    steps = 25,
):
    pipeline, tokenizer = load_models(pretrained_model_name_or_path, device)

    root_dir = root_dir + f'-{num_frames}-action'

    os.makedirs(root_dir, exist_ok=True)

    print('loaded pipeline!')
    
    print('loading dataset ing...')
    train_dataset = Actionframes(None, split='val', tokenizer=tokenizer, img_size=(192, 384), max_video_len = num_frames)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=8,
        drop_last=False
    )

    paths = pretrained_model_name_or_path.split('/')
    base_model = paths[-2]
    checkpoint = paths[-1]

    version = base_model + f"-{checkpoint}" + f'-{cfg_min}-{cfg_max}-step{steps}'

    os.makedirs(os.path.join(root_dir, version), exist_ok=True)

    # caption model
    git_processor_large = AutoProcessor.from_pretrained("/mnt/lustrenew/wangxiaodong/models/git-large-coco")
    git_model_large = AutoModelForCausalLM.from_pretrained("/mnt/lustrenew/wangxiaodong/models/git-large-coco")
    print('loaded caption model!')

    new_batch = []

    imgarr = None
    for n, batch in tqdm(enumerate(train_dataloader)):

        video = pipeline(batch, num_frames=train_frames, height=192, width=384, min_guidance_scale=cfg_min, max_guidance_scale=cfg_max, num_inference_steps=steps).frames

        names = batch['name']
        scenes = batch['scene']
        imgarr = batch['pil']

        # print(type(video)) #list

        idx = 0
        for r in range(1, roll_out):
            idx = idx + 7
            curr_batch = {}
            for k, v in batch.items():
                if 'steer' in k or 'speed' in k:
                    curr_batch[k] = v[:, idx:]
                else:
                    curr_batch[k] = v
            curr_frame = video[0][-1]
            # print(f'curr: {curr_frame}')
            curr_prompt = generate_caption(curr_frame, git_processor_large, git_model_large)

            video_ex = pipeline(curr_batch, image=[curr_frame], prompt=[curr_prompt], num_frames=train_frames, height=192, width=384, min_guidance_scale=cfg_min, max_guidance_scale=cfg_max, num_inference_steps=steps).frames
            video[0] = video[0] + video_ex[0][1:]
        print(f'len of final video {len(video)}')
        
        
        for i in range(len(names)):
            name = os.path.basename(names[i]).split('.')[0]
            os.makedirs(os.path.join(root_dir, version, scenes[i]), exist_ok=True)
            im = Image.fromarray(imgarr[i].numpy().astype('uint8')).convert('RGB')
            # im.save(os.path.join(root_dir, version, scenes[i], f'{name}.jpg'))
            # NOTE if use CFG, you should replace the first frame with ground truth
            if cfg_max > 1:
                video[i][0] = im.resize((384, 192))
            export_to_video(video[i], os.path.join(root_dir, version, scenes[i], f'{name}.mp4'), fps=6)

    print('inference done!')

if __name__ == '__main__':
    fire.Fire(main)