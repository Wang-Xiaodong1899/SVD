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

# from diffusers.models.action_layers_norm import ActionCoarseModule
from diffuser.models.action_layers_norm import ActionAttnModule


def load_models(ckpt="/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels-vis/action-vae-5e-5/checkpoint-3000", device='cuda:0'):
    vae = AutoencoderKL.from_pretrained(
                '/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels/image-ep50-s192-1e-4', subfolder="vae"
    )
    vae.eval()
    vae.to(device)
    tokenizer = CLIPTokenizer.from_pretrained('/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels/image-ep50-s192-1e-4', subfolder="tokenizer")

    action_model = ActionAttnModule(cross_attention_dim=640)
    action_model.eval()
    action_model = action_model.to(device)
    action_model.load_state_dict(torch.load(os.path.join(ckpt,'ckpt.pth'), map_location='cpu'))

    print(f'loaded weight from {ckpt}')
    
    return action_model, vae, tokenizer

def numpy_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    raise TypeError(f"{type(obj)} is not JSON serializable")

def denormalize_func(normalized_tensor, min_val=0, max_val=200):
    tensor = (normalized_tensor + 1) / 2
    tensor = tensor * (max_val - min_val) + min_val
    # tensor = t.round(tensor).long()
    return tensor



def main(
    ckpt = '/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/smodels-vis/action-vae-Attn-5e-5/checkpoint-6500',
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

            steers = batch['steer_inp'] # b, f
            speeds = batch['speed_inp'] # b, f

            steers_tgt = batch['steer_tgt']
            speeds_tgt = batch['speed_tgt']

            # action_gt = torch.stack([steers_gt, speeds_gt], dim=-1) # b, f, 2
            action = torch.stack([steers, speeds], dim=-1) # b, f, 2

            history_action = action[:, :history_len]
            # future_action = action_gt[:, history_len:]
            # history_action_gt = action_gt[:, :history_len]

            history_action = history_action.to(device)

            action_pred = action_model(history_action, latents)

            # denorm
            steers_tgt = denormalize_func(steers_tgt, -9, 9)
            speeds_tgt = denormalize_func(speeds_tgt, 0, 70)

            his_gt_steer = steers_tgt[:, :history_len]
            his_gt_speed = speeds_tgt[:, :history_len]

            fut_gt_steer = steers_tgt[:, history_len:]
            fut_gt_speed = speeds_tgt[:, history_len:]

            steer_pred = action_pred[..., 0]
            speed_pred = action_pred[..., 1]
            steer_pred = denormalize_func(steer_pred, -9, 9)
            speed_pred = denormalize_func(speed_pred, 0, 70)

            his_gt_action = torch.stack([his_gt_steer, his_gt_speed], dim=-1)
            fut_gt_action = torch.stack([fut_gt_steer, fut_gt_speed], dim=-1)
            pred_action = torch.stack([steer_pred, speed_pred], dim=-1)

            for idx_n, name in enumerate(names):
                action_hat_sam = pred_action[idx_n]
                action_gt_sam = fut_gt_action[idx_n]
                action_his_sam = his_gt_action[idx_n]
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