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

from diffusers.pipelines.stable_video_diffusion.pipeline_action_video_diffusion_v1 import ActionVideoDiffusionPipeline
from diffusers.utils import export_to_video

from fvd import load_fvd_model, compute_fvd, compute_fvd_1

DATAROOT = '/mnt/lustrenew/wangxiaodong/data/nuscene'

import json


def main(
        gt_dir='/ssd_datasets/wxiaodong/nuscene_val/val_150_video/',
        tgt_dir='/mnt/lustrenew/wangxiaodong/data/nuscene/FVD-first-15/video-v11-ep200-s196',
        version = None,
        split='val',
        num_frames=8,
        i3d_device='cuda:0',
        eval_frames=8,
):

    if version is None:
        version = os.path.basename(tgt_dir)+f'_F{num_frames}'
    
    print(f'eval videos at {tgt_dir}')
    # load fvd model
    i3d_model = load_fvd_model(i3d_device)
    print('loaded i3d_model')

    # read real video
    meta_data = json2data('/mnt/lustrenew/wangxiaodong/data/nuscene/samples_group_sort_val.json')
    files_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/val_group'

    # read syn videos
    syn_videos = []
    # video_files = os.listdir(tgt_dir)
    for item in tqdm(meta_data):
        sce = item['scene']
        files = os.listdir(os.path.join(tgt_dir, sce))
        for file in files:
            if file.split('.')[-1] == 'mp4':
                video_arr = mp4toarr(os.path.join(tgt_dir, sce, file))
                syn_videos.append(video_arr[:eval_frames]) # only load eval_frames
    syn_videos = np.array(syn_videos)
    print(f'syn shape {syn_videos.shape}')

    # real_videos = []
    # for item in tqdm(meta_data):
    #     sce = item['scene']
    #     frames_info = item['samples']
    #     frames = []
    #     for im_path in frames_info[:eval_frames]:
    #         im = image2pil(os.path.join(files_dir, sce, os.path.basename(im_path)))
    #         frames.append(im)
    #     frames = pil2arr(frames)
    #     frames = frames[:eval_frames] # only load eval_frames
    #     real_videos.append(frames)
    # # N T H W C
    # real_videos = np.array(real_videos)
    # print(f'real shape {real_videos.shape}')

    # fvd_score = compute_fvd(real_videos, syn_videos, i3d_model, i3d_device)
    fvd_score = compute_fvd_1(syn_videos, syn_videos, i3d_model, i3d_device)
    
    print('Save Done!')

    print(f'{version} AVG FVD: {fvd_score}')

if __name__ == '__main__':
    fire.Fire(main)