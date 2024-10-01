import sys
sys.path.append('/workspace/wxd/SVD/src')

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

from diffusers.utils import export_to_video

from fvd import load_fvd_model, compute_fvd, compute_fvd_1


import json


def main(
        gt_dir='/mnt/storage/user/wangxiaodong/nuscenes/val_video',
        tgt_dir='/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/FVD-allframe-first-85-rec/ti2v_s448_imclip-checkpoint-16000',
        version = None,
        split='val',
        num_frames=8,
        i3d_device='cuda:0',
        eval_frames=16,
        skip_gt = 0,
        skip_pred = 0
):

    if version is None:
        version = os.path.basename(tgt_dir)+f'_F{num_frames}'
    
    print(f'eval videos at {tgt_dir}')
    # load fvd model
    i3d_model = load_fvd_model(i3d_device)
    print('loaded i3d_model')

    # read real video
    meta_data = json2data('/mnt/storage/user/wangxiaodong/nuscenes/val_video/scenes.json')
    # files_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/val_group'

    # read syn videos
    syn_videos = []
    for sce in tqdm(meta_data):
        files = os.listdir(os.path.join(tgt_dir, sce))
        for file in files:
            if file.split('.')[-1] == 'mp4':
                video_arr = mp4toarr(os.path.join(tgt_dir, sce, file))
                # syn_videos.append(video_arr[:eval_frames]) # only load eval_frames

                # NOTE random choose 6 times
                # choosed_idxs = random.sample(range(len(video_arr)-eval_frames), 6)
                
                # choosed_idxs = range(6)
                # for idx in choosed_idxs:
                # syn_videos.append(video_arr[::4][:eval_frames])
                # syn_videos.append(video_arr[::3][:eval_frames])
                for i in range(6):
                    syn_videos.append(video_arr[i:][::5][:eval_frames])
                for i in range(6):
                    syn_videos.append(video_arr[i:][::4][:eval_frames])
                # syn_videos.append(video_arr[::3][:eval_frames])

    syn_videos = np.array(syn_videos)
    print(f'syn shape {syn_videos.shape}')

    real_videos = []
    for sce in tqdm(meta_data):
        files = os.listdir(os.path.join(gt_dir, sce))
        for file in files:
            if file.split('.')[-1] == 'mp4':
                video_arr = mp4toarr(os.path.join(gt_dir, sce, file))
                # sample_frames = video_arr[::skip_gt+1]
                # if len(sample_frames)<eval_frames:
                #     sample_frames = list(sample_frames) + [sample_frames[-1]] * (eval_frames-len(sample_frames))
                #     print(len(sample_frames), eval_frames-len(sample_frames))

                # real_videos.append(sample_frames[:eval_frames]) # only load eval_frames
                choosed_idxs = random.sample(range(len(video_arr)-eval_frames), 6)
                for idx in choosed_idxs:
                    real_videos.append(video_arr[idx: idx+eval_frames])
                
                # real_videos.append(video_arr[0: eval_frames])


    # N T H W C
    real_videos = np.array(real_videos)
    print(f'real shape {real_videos.shape}')

    fvd_score = compute_fvd(real_videos, syn_videos, i3d_model, i3d_device)
    # fvd_score = compute_fvd_1(real_videos, syn_videos, i3d_model, i3d_device)
    
    print('Save Done!')

    print(f'{version} AVG FVD: {fvd_score}')

if __name__ == '__main__':
    fire.Fire(main)