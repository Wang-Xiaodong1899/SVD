#TODO load nuscene data and load model to inference the image caption
import os
import torch
import torchvision
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
import fire

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

import sys
sys.path.append('/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/src')

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from transformers import AutoProcessor, AutoModelForCausalLM

DATAROOT = '/mnt/storage/user/wangxiaodong/nuscenes'
img_size = (192, 384)

def load_models(pretrained_model_name_or_path = '/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100', unet_path='', device='cuda:0'):
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

    unet_path = unet_path if unet_path is not None else pretrained_model_name_or_path+'/unet'

    unet = UNet2DConditionModel.from_pretrained(unet_path)
    unet.eval()
    unet = unet.to(device)
    pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            safety_checker=None
    )
    pipeline = pipeline.to(device)
    return pipeline


class val_dataset(torch.utils.data.Dataset):
    def __init__(self, split='val', use_scene_desc=False):
        super().__init__()
        self.split = split
        self.use_scene_desc = use_scene_desc

        # suppose to save original image
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)

        self.splits = create_splits_scenes()

        # training samples
        self.samples = self.get_samples(split)

        # read from json
        # json_path = '/ssd_datasets/wxiaodong/nuscene/nuscene_caption_val.json'
        # with open(json_path, 'r') as f:
        #     self.samples = json.load(f)

        json_path = f'/mnt/storage/user/wangxiaodong/nuscenes/nuscene_caption_utime_{split}.json'
        with open(json_path, 'r') as f:
            self.caption_utime = json.load(f)
        print('Total samples: %d' % len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def get_samples(self, split='val'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def __getitem__(self, index):
        try:
            my_sample = self.samples[index]

            # only front camera
            sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
            file_path = sample_data['filename']
            utime = file_path.split('__')[-1].split('.')[0]

            caption = self.caption_utime[utime]

            image_path = os.path.join(DATAROOT, file_path)
            # my_sample = self.samples[index]
            # file_path = my_sample['path']
            # caption = my_sample['caption']

            if self.use_scene_desc:
                sample_token = sample_data["sample_token"]
                scene_token = self.nusc.get('sample', sample_token)["scene_token"]
                caption = self.nusc.get('scene', scene_token)["description"]
            else:
                # NOTE test no-text version
                caption = ""

            return {
                'path': image_path, # to read and save to groun-truth directory
                'caption': caption,
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

def main(
        base_model='/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100',
        unet_path="/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100/checkpoint-10000",
        version = None,
        split='val',
        device='cuda:0',
        save_gt=False,
        save_dir='/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/results',
        size=(192, 384),
        use_scene_desc=False
):
    os.makedirs(save_dir, exist_ok=True)
    if version==None:
        version = os.path.basename(base_model).split('.')[0]
    print(f'synthsizing image for version {version}')
    if save_gt:
        os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, version), exist_ok=True)

    pipeline = load_models(base_model, unet_path).to(device).to(torch.float16)

    print(f'synthsizing image from {split} split')
    dataset = val_dataset(split=split, use_scene_desc=use_scene_desc)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, drop_last=False)

    captions = []

    for i, batch in tqdm(enumerate(dataloader)):
        caption = batch['caption']
        path = batch['path']
        images = pipeline(caption, height=size[0], width=size[1], num_inference_steps=25, guidance_scale=1).images
        
        #save gt image
        if save_gt:
            for p in path:
                name = os.path.basename(p)
                im = Image.open(p).convert('RGB').resize((384, 192))
                im.save(os.path.join(save_dir, 'gt', name))
        for (idx, p) in enumerate(path):
            ims = images[idx]
            name = os.path.basename(p)
            ims.save(os.path.join(save_dir, version, name))
    
    print('Save Done!')

if __name__ == '__main__':
    # unet = UNet2DConditionModel.from_pretrained("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100/unet")
    # unet.eval()
    # unet.to(torch.float16)
    # unet.save_pretrained("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100/unet-fp16")
    fire.Fire(main)