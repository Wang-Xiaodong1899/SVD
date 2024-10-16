import sys
sys.path.append('/mnt/cache/wangxiaodong/SDM/src')

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import random
import torch
import transformers
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import copy
import os
from tqdm import tqdm
from einops import repeat
import fire
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


# validate FVD on Nuscene val 150 samples
# default 8 frames
# default image size 256x256

# NOTE:
# to ensure accurate
# we load the video following the time steps

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from common import image2arr, pil2arr, mp4toarr, image2pil, json2data

# from diffusers.models.unet_action import UNetSpatioTemporalConditionModel_Action
from diffusers.models.unet_action_v11 import UNetSpatioTemporalConditionModel_Action
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from diffusers.pipelines.stable_video_diffusion.pipeline_action_video_diffusion_v1 import ActionVideoDiffusionPipeline
from diffusers.utils import export_to_video

from fvd import load_fvd_model, compute_fvd

DATAROOT = '/ssd_datasets/wxiaodong/nuscene'

import json


def main(
        gt_dir='/ssd_datasets/wxiaodong/nuscene_val/val_150_video/',
        tgt_dir='/ssd_datasets/wxiaodong/nuscene_val/FVD-center-8/drive-video-s256-v01-ep100',
        version = None,
        split='val',
        num_frames=8,
        i3d_device='cuda:0',
        eval_frames=8,
):

    if version is None:
        version = os.path.basename(tgt_dir)+f'_F{num_frames}'
    # load fvd model
    i3d_model = load_fvd_model(i3d_device)
    print('loaded i3d_model')

    # read real video
    meta_data = json2data('/home/wxd/data/nuscene/samples_group_sort_val.json')
    files_dir = '/home/wxd/data/nuscene/val_group'

    real_videos = []
    for item in tqdm(meta_data):
        sce = item['scene']
        frames_info = item['samples']
        frames = []
        for im_path in frames_info:
            im = image2pil(os.path.join(files_dir, sce, os.path.basename(im_path)))
            frames.append(im)
        frames = pil2arr(frames)
        frames = frames[:eval_frames] # only load eval_frames
        real_videos.append(frames)
    # N T H W C
    real_videos = np.array(real_videos)
    print(f'real shape {real_videos.shape}')

    # read syn videos
    syn_videos = []
    video_files = os.listdir(tgt_dir)
    for video_path in tqdm(video_files):
        if video_path.split('.')[-1] != 'mp4':
            continue
        video_arr = mp4toarr(os.path.join(tgt_dir, video_path))
        syn_videos.append(video_arr[:eval_frames]) # only load eval_frames
    syn_videos = np.array(syn_videos)
    print(f'syn shape {syn_videos.shape}')

    fvd_score = compute_fvd(real_videos, syn_videos, i3d_model, i3d_device)
    
    print('Save Done!')

    print(f'{version} AVG FVD: {fvd_score}')

if __name__ == '__main__':
    fire.Fire(main)