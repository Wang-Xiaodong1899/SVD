import sys
sys.path.append('/home/wxd/video-generation/diffusers/src')

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

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from common import image2arr, pil2arr

from diffusers.models.unet_action import UNetSpatioTemporalConditionModel_Action
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from diffusers.pipelines.stable_video_diffusion.pipeline_action_video_diffusion_v1 import ActionVideoDiffusionPipeline
from diffusers.utils import export_to_video

from fvd import load_fvd_model, compute_fvd

DATAROOT = '/ssd_datasets/wxiaodong/nuscene'


# default image size 256x512

class Videoframes(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='val', max_video_len = 8, img_size=(256,512)):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        self.max_video_len = max_video_len
        self.action_scale = 1

        # text tokenizer

        # read from json
        json_path = f'/ssd_datasets/wxiaodong/nuscene/frame_action_{split}.json'
        with open(json_path, 'r') as f:
            self.frame_action = json.load(f)
        
        # scenes num
        self.scenes = list(self.frame_action.keys())
        print('Total samples: %d' % len(self.scenes))

        # utime-caption
        json_path = f'/ssd_datasets/wxiaodong/nuscene/nuscene_caption_utime_{split}.json'
        with open(json_path, 'r') as f:
            self.caption_utime = json.load(f)
    
    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, index):
        try:
            my_scene = self.scenes[index]

            scene_data = self.frame_action[my_scene] #[ [steer], [speed], [path] ]
            inner_frame_len = len(scene_data[0]) # maybe 40

            # center start, fix here
            seek_start = (inner_frame_len - self.max_video_len)//2

            # add frame
            steers = scene_data[0]
            speeds = scene_data[1]
            paths = scene_data[2]

            seek_steer = steers[seek_start: seek_start+self.max_video_len]
            seek_speed = speeds[seek_start: seek_start+self.max_video_len]
            seek_path = paths[seek_start: seek_start+self.max_video_len]

            # align and pad 0
            seek_steer = [0.] + seek_steer[1:]
            seek_speed = [0.] + seek_speed[1:]

            # transfer
            seek_steer = torch.tensor(seek_steer) * np.pi / 180 * self.action_scale
            seek_speed = torch.tensor(seek_speed) * 10. / 36 * self.action_scale

            first_image_path = seek_path[0]
            utime = first_image_path.split('__')[-1].split('.')[0]
            first_image_caption = self.caption_utime[utime]

            frame_paths = [os.path.join(DATAROOT, file_path) for file_path in seek_path]
            video = np.stack([image2arr(fn) for fn in frame_paths])
            image = video[0]
            video = video.astype(np.uint8)

            imgs = video[:self.max_video_len]

            # suppose use caption for first sample
            # inputs = self.tokenizer(
            #     first_image_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            # )
            # inputs_id = inputs['input_ids'][0] # no question

            return {
                'caption': first_image_caption,
                'label_imgs': imgs, # np type
                'image': image,
                'steer': seek_steer,
                'speed': seek_speed,
                'scene': my_scene,
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))
        
def main(
        unet_model='/ssd_datasets/wxiaodong/ckpt/drive-video-s256-v01-ep100',
        base_model='/ssd_datasets/wxiaodong/ckpt/drive-video-s256-v01-ep100',
        version = None,
        split='val',
        device='cuda:7',
        save_gt=False,
        save_dir='/ssd_datasets/wxiaodong/nuscene_val/FVD_val',
        num_frames=8,
        i3d_device='cuda:6',
):
    os.makedirs(save_dir, exist_ok=True)
    if version==None:
        version = os.path.basename(base_model).split('.')[0]
    print(f'synthsizing image for version {version}')
    if save_gt:
        os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, version), exist_ok=True)

    pretrained_model_name_or_path = base_model
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

    # The U-Net Path
    unet = UNetSpatioTemporalConditionModel_Action.from_pretrained(unet_model, subfolder="unet")
    unet.eval()
    unet.to(device)

    pipeline = ActionVideoDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor
)
    pipeline.to(device)

    # load fvd model
    i3d_model = load_fvd_model(i3d_device)


    print(f'synthsizing Video from {split} split')
    dataset = Videoframes(split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=16, drop_last=False)

    captions = []

    avg_fvd = 0.

    for i, batch in tqdm(enumerate(dataloader)):
        caption = batch['caption']
        bsz = len(caption)
        scene_name = batch['scene']
        images = batch['image']
        real_ = batch['label_imgs'] #b t h w c
        
        frames = pipeline(images, num_frames=num_frames, prompt=caption, action=None, height=256, width=512).frames[0]
        # return a list [[f1,f2,f3],[f1,f2,f3]]

        #replace first frame
        syn_videos = []
        videos = []
        for vidx in range(bsz):
            video = frames[vidx]
            video[0] = images[vidx] # replace first frame
            videos.append(video) # for saving
            syn_video = pil2arr(video) # to b t h w c np.uint8, for i3d
            syn_videos.append(syn_video)
        syn_videos = np.stack(syn_videos)

        fvd_score = compute_fvd(real_, syn_videos, i3d_model, i3d_device)
        avg_fvd += fvd_score

        #save gt image
        if save_gt:
            for (idx, sce) in enumerate(scene_name):
                real = real_[idx] # batch ->instance
                real = [frame for frame in real]
                export_to_video(real, os.path.join(save_dir, 'gt', sce), fps=6) #at least 6
        for (idx, scene_name) in enumerate(scene_name):
            video = videos[idx]
            export_to_video(video, os.path.join(save_dir, version, sce), fps=6) # at least 6
    
    print('Save Done!')

    print(f'AVG FVD: {avg_fvd/len(dataloader)}')

if __name__ == '__main__':
    fire.Fire(main)