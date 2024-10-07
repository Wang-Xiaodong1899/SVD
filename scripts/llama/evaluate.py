# NOTE
# This file is for aliyun 1-GPU test

import sys
sys.path.append('/workspace/wxd/SVD/src')
sys.path.append('/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug')

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

from diffuser.utils import export_to_video

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler

from diffuser.models.unet_dwm import UNetSpatioTemporalConditionModel_Action
from diffuser.pipelines.worldmodel.pipeline import ActionVideoDiffusionPipeline

from nuscene_action_dwm_evaluate import Actionframes
from scripts.llama.action2video_dwm import WorldModel
from llama.attention_ngram_dwm import LlamaModeling

import numpy as np
import json

def numpy_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    raise TypeError(f"{type(obj)} is not JSON serializable")


def load_models(pretrained_model_name_or_path = '/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels-vis/wm_action2video_freezeactionmodel_bs32/checkpoint-15000/', device='cuda:0', zero_pad=False):
    text_encoder = CLIPTextModel.from_pretrained(
                '/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100', subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
                '/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100', subfolder="vae"
    )
    clip_model = CLIPModel.from_pretrained(
        "/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/clip-vit-large-patch14", torch_dtype=torch.float16)

    text_encoder.eval()
    vae.eval()
    clip_model.eval()
    text_encoder.to(device)
    vae.to(device)
    clip_model.to(device)

    tokenizer = CLIPTokenizer.from_pretrained('/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100', subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained('/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100', subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained('/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100', subfolder="feature_extractor")

    unet = UNetSpatioTemporalConditionModel_Action(cross_attention_dim=768, in_channels=4, temp_style = "image", zero_pad=zero_pad)
    if "wm_ckpt.pth" in os.listdir(pretrained_model_name_or_path):
        tensors = torch.load(os.path.join(pretrained_model_name_or_path, 'wm_ckpt.pth'), map_location="cpu")
    else:
        if "unet" not in os.listdir(pretrained_model_name_or_path):
            dm_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors")
        else:
            dm_path = os.path.join(pretrained_model_name_or_path, "unet", "diffusion_pytorch_model.safetensors")
        tensors = {}
        with safe_open(dm_path, framework="pt", device='cpu') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    
    unet_state_dict = {}
    action_model_state_dict = {}

    # Iterate through the keys in the full state_dict
    for key, value in tensors.items():
        # Check if the key belongs to UNet or action_model
        if key.startswith('unet'):
            unet_state_dict[key[5:]] = value
        elif key.startswith('action_model'):
            action_model_state_dict[key[13:]] = value
        else:
            # Handle other keys as needed
            pass
    miss_keys, ignore_keys = unet.load_state_dict(unet_state_dict, strict=False)
    print(f'loaded unet from {pretrained_model_name_or_path}')
    print('miss_keys: ', miss_keys)
    print('ignore_keys: ', ignore_keys)

    # initialize action model
    from evaluate_config import LlamaConfig
    config = LlamaConfig()
    action_model = LlamaModeling(config)
    miss_keys, ignore_keys = action_model.load_state_dict(action_model_state_dict, strict=False)
    print(f'loaded unet from {pretrained_model_name_or_path}')
    print('miss_keys: ', miss_keys)
    print('ignore_keys: ', ignore_keys)
    action_model.eval()
    action_model.to(device)

    unet.eval()
    unet = unet.to(device)
    pipeline = ActionVideoDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            clip_model=clip_model,
            action_model=action_model
    )
    pipeline = pipeline.to(device).to(torch.float16)
    return pipeline, tokenizer, config

def generate_caption(image, git_processor_large, git_model_large, device='cuda:0'):

    git_model_large.to(device)
    pil_list = [image.convert('RGB')]
    inputs = git_processor_large(images=pil_list, return_tensors="pt").to(device)
    
    generated_ids = git_model_large.generate(pixel_values=inputs.pixel_values, max_length=50)

    generated_caption = git_processor_large.batch_decode(generated_ids, skip_special_tokens=True)
   
    return generated_caption[0]

# /mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels-vis/wm_action2video_ac_int1_ctx8_bs16_run1/checkpoint-15000
# /mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels-vis/wm_action2video_ac_int1_ctx4_bs32/checkpoint-40000
# /mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels-vis/wm_action2video_freezeac_ctx4_bs32/worldmodel


def main(
    pretrained_model_name_or_path = '/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels-vis/wm_action2video_freezeactionmodel_bs32/checkpoint-15000',
    num_frames = 36,
    root_dir = '/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/FVD-first',
    train_frames = 8,
    device='cuda',
    roll_out= 1,
    cfg_min = 1.0,
    cfg_max = 3.0,
    steps = 25,
    debug = False,
    debug_frames = 36,
    times = 1,
    random_ref_img = False,
    ctx_frame = 1,
    zero_pad = False
):
    pipeline, tokenizer, config = load_models(pretrained_model_name_or_path, device, zero_pad)

    root_dir = root_dir + f'-{debug_frames}-action-{times}-ctxFrame-{ctx_frame}'

    if debug:
        root_dir = root_dir + "-debug2"

    os.makedirs(root_dir, exist_ok=True)

    print('loaded pipeline!')
    
    print('loading dataset ing...')
    train_dataset = Actionframes(config, split='val', tokenizer=tokenizer, img_size=(192, 384), max_video_len = num_frames, random_ref_img=random_ref_img)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=8,
        drop_last=False
    )
    if pretrained_model_name_or_path[-1] == '/':
        pretrained_model_name_or_path = pretrained_model_name_or_path[:-1]
    paths = pretrained_model_name_or_path.split('/')
    base_model = paths[-2]
    checkpoint = paths[-1]

    version = base_model + f"-{checkpoint}" + f'-{cfg_min}-{cfg_max}-step{steps}'

    os.makedirs(os.path.join(root_dir, version), exist_ok=True)

    # caption model
    git_processor_large = AutoProcessor.from_pretrained("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/git-large-coco")
    git_model_large = AutoModelForCausalLM.from_pretrained("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/git-large-coco")
    print('loaded caption model!')

    new_batch = []

    imgarr = None
    for n, batch in tqdm(enumerate(train_dataloader)):

        if debug:
            if n>2:
                break

        steers = batch['steer'] # b, f
        speeds = batch['speed'] # b, f
        action = torch.stack([steers, speeds], dim=-1) # b, f, 2 cpu device

        action_gt = action
        action_pred = [action[:, :config.history_len]] # preverse history action

        # cut for src_action
        action = action[:, :config.history_len] # b 4 2

        videos = None
        last_frame = []
        prompt = None

        # NOTE
        # 3 -> 22
        # 5 -> 36 (ctx=1)
        # 8 -> 36 (ctx=4)
        # 6 -> 38 (ctx=2)
        rollout = 5
        if ctx_frame == 4:
            rollout = 8
        elif ctx_frame == 2:
            rollout = 6
        for ac_idx in tqdm(range(rollout)): # 4+4+4

            # ac_idx does not need action_only
            if ac_idx > 0:
                # action predict only
                action_hat = pipeline(batch, prompt=prompt, image=last_frame, num_frames=train_frames, height=192, width=384, min_guidance_scale=cfg_min, max_guidance_scale=cfg_max, num_inference_steps=steps, action=action, action_start=0, action_end=ac_idx*2*config.n_gram, action_only=True, ctx_frame=ctx_frame).actions
                action_pred.append(action_hat.cpu())
                # action + action_pred
                action = torch.cat([action, action_hat.cpu()], dim=1)
            
            if ac_idx > 0:
                ac_idx = ac_idx * 2 # action only n_gram + n_gram
            outputs = pipeline(batch, prompt=prompt, image=last_frame, num_frames=train_frames, height=192, width=384, min_guidance_scale=cfg_min, max_guidance_scale=cfg_max, num_inference_steps=steps, action=action, action_start=0, action_end=ac_idx*config.n_gram+config.n_gram, ctx_frame=ctx_frame)
            print(f'last frame: {len(last_frame)}')
            video = outputs.frames
            action_hat = outputs.actions

            # import pdb
            # pdb.set_trace()

            if videos is None:
                videos = video
            else:
                videos[0] = videos[0] + video[0][ctx_frame:]

            last_frame = video[0][-1]
            prompt = generate_caption(last_frame, git_processor_large, git_model_large)
            # last_frame = [last_frame]
            last_frame = video[0][-ctx_frame:]

            action_pred.append(action_hat.cpu())
            # action + action_pred
            action = torch.cat([action, action_hat.cpu()], dim=1)
        
        action_pred = torch.cat(action_pred, dim=1) # b l d

        print(f'len of final video {len(videos[0])}')
        print(f'len og final action {len(action_pred)}')


        names = batch['name']
        scenes = batch['scene']
        imgarr = batch['pil']
        imgarrs = batch['pils']
        captions = batch['captions']
        # print(type(video)) #list

        # idx = 0
        # for r in range(1, roll_out):
        #     idx = idx + 7
        #     curr_batch = {}
        #     for k, v in batch.items():
        #         if 'steer' in k or 'speed' in k:
        #             curr_batch[k] = v[:, idx:]
        #         else:
        #             curr_batch[k] = v
        #     curr_frame = video[0][-1]

        #     # print(f'curr: {curr_frame}')
        #     # curr_prompt = generate_caption(curr_frame, git_processor_large, git_model_large)
        #     # print(captions)
        #     curr_prompt = captions[idx][0]

        #     print(curr_prompt)

        #     video_ex = pipeline(curr_batch, image=[curr_frame], prompt=[curr_prompt], num_frames=train_frames, height=192, width=384, min_guidance_scale=cfg_min, max_guidance_scale=cfg_max, num_inference_steps=steps).frames
        #     video[0] = video[0] + video_ex[0][1:]
        # print(f'len of final video {len(video)}')
        
        
        for i in range(len(names)):
            name = os.path.basename(names[i]).split('.')[0]
            os.makedirs(os.path.join(root_dir, version, scenes[i]), exist_ok=True)
            im = Image.fromarray(imgarr[i].numpy().astype('uint8')).convert('RGB')
            ims = [Image.fromarray(item.numpy().astype('uint8')).convert('RGB') for item in imgarrs[i]]
            # im.save(os.path.join(root_dir, version, scenes[i], f'{name}.jpg'))
            # NOTE if use CFG, you should replace the first frame with ground truth
            # if cfg_max > 1:
            # videos[i][0] = im.resize((384, 192))
            for ctx in range(ctx_frame):
                videos[i][ctx] = ims[ctx].resize((384, 192))
            export_to_video(videos[i], os.path.join(root_dir, version, scenes[i], f'{name}.mp4'), fps=6)

        # NOTE save actions
        action_pred = action_pred.cpu().numpy()
        action_gt = action_gt.cpu().numpy()

        outputs = {
            'action_pred': action_pred,
            'action_gt': action_gt
        }

        with open(os.path.join(root_dir, version, scenes[i], 'action.json'), 'w') as f:
            json.dump(outputs, f, default=numpy_serializable)

    print('inference done!')

if __name__ == '__main__':
    fire.Fire(main)