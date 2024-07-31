# NOTE
# This file is for aliyun 1-GPU test

import sys
sys.path.append('/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/src')
sys.path.append('/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD')

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
import json
import numpy as np

from diffusers.utils import export_to_video

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler

from scripts.joint_output.nuscene_action_val_ms import Actionframes

from transformers import AutoProcessor, AutoModelForCausalLM

from diffusers.models.action_layers_norm import ActionCoarseModule


def load_models(ckpt="/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels-vis/action-vae-5e-5/checkpoint-3000", device='cuda:0'):
    vae = AutoencoderKL.from_pretrained(
                '/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels/image-ep50-s192-1e-4', subfolder="vae"
    )
    vae.eval()
    vae.to(device)
    tokenizer = CLIPTokenizer.from_pretrained('/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels/image-ep50-s192-1e-4', subfolder="tokenizer")

    action_model = ActionCoarseModule(cross_attention_dim=640)
    action_model.eval()
    action_model = action_model.to(device)
    action_model.load_state_dict(torch.load(os.path.join(ckpt,'ckpt.pth'), map_location='cpu'))

    print(f'loaded weight from {ckpt}')
    
    return action_model, vae, tokenizer

def numpy_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    raise TypeError(f"{type(obj)} is not JSON serializable")


def main(
    ckpt = '/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels-vis/action-vae-5e-5-reproduce-ms/checkpoint-13500',
    num_frames = 8,
    root_dir = '/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/FVD-first',
    train_frames = 8,
    device='cuda',
    roll_out= 1,
    cfg_min = 1.0,
    cfg_max = 1.0,
    steps = 25,
    history_len = 8,
):
    action_model, vae, tokenizer = load_models(ckpt, device)
    print('loaded pipeline!')

    root_dir = root_dir + f'-{num_frames}-action'

    os.makedirs(root_dir, exist_ok=True)
    
    print('loading dataset ing...')
    train_dataset = Actionframes(None, split='val', tokenizer=tokenizer, img_size=(192, 384), max_video_len = num_frames)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=8,
        drop_last=False
    )

    paths = ckpt.split('/')
    base_model = paths[-2]
    checkpoint = paths[-1]

    version = base_model + f"-{checkpoint}" + f'-debug'

    os.makedirs(os.path.join(root_dir, version), exist_ok=True)

    new_batch = []

    imgarr = None
    action_compare = []
    for n, batch in tqdm(enumerate(train_dataloader)):
        # if n>10:
        #     break
        names = batch['name']
        scenes = batch['scene']
        imgarr = batch['pil']
        im_tensor = batch['image']

        with torch.no_grad():
            im_tensor = im_tensor.to(device)

            latents = vae.encode(im_tensor).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            steers = batch['steer'] # b, f
            speeds = batch['speed'] # b, f

            action = torch.stack([steers, speeds], dim=-1) # b, f, 2

            history_action = action[:, :history_len]
            future_action = action[:, history_len:]

            history_action = history_action.to(device)

            action_pred = action_model(history_action, latents)

            for idx_n, name in enumerate(names):
                action_hat_sam = action_pred[idx_n]
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
        
    with open(os.path.join(root_dir, version, 'action.json'), 'w') as f:
        json.dump(action_compare, f, default=numpy_serializable)
    
    print('inference done!')

if __name__ == '__main__':
    fire.Fire(main)