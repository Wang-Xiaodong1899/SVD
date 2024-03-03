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

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

import sys
sys.path.append('/mnt/cache/wangxiaodong/SDM/src')

from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, UNet2DConditionModel

DATAROOT = '/mnt/lustrenew/wangxiaodong/data/nuscene'
img_size = (384, 768)


class val_dataset(torch.utils.data.Dataset):
    def __init__(self, split='val'):
        super().__init__()
        self.split = split

        # suppose to save original image

        # read from json
        json_path = '/mnt/lustrenew/wangxiaodong/data/nuscene/nuscene_caption_val.json'
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        print('Total samples: %d' % len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        try:
            my_sample = self.samples[index]
            file_path = my_sample['path']
            caption = my_sample['caption']

            return {
                'path': file_path, # to read and save to groun-truth directory
                'caption': caption,
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

def main(
        base_model='/mnt/lustrenew/wangxiaodong/models/stable-diffusion-2-1',
        version = None,
        split='val',
        device='cuda:0',
        save_gt=False,
        save_dir='/mnt/lustrenew/wangxiaodong/data/nuscene_val',
        unet_path='/mnt/lustrenew/wangxiaodong/smodels/image-v-s768-1e-4/checkpoint-4000'
):
    os.makedirs(save_dir, exist_ok=True)
    if version==None:
        version = os.path.basename(base_model).split('.')[0]
    if unet_path is not None and 'checkpoint' in unet_path:
        version = unet_path.split('/')[-2] + '-' + unet_path.split('/')[-1]
    print(f'synthsizing image for version {version}')
    if save_gt:
        os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, version), exist_ok=True)

    # all components
    text_encoder = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", variant='fp16'
    )
    vae = AutoencoderKL.from_pretrained(
        base_model, subfolder="vae", variant='fp16'
    )

    text_encoder.eval()
    vae.eval()
    text_encoder.to(device)
    vae.to(device)

    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")

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
    pipeline= pipeline.to(device).to(torch.float16)

    print(f'synthsizing image from {split} split')
    dataset = val_dataset(split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)

    captions = []

    print(f'syn images at resolution: {img_size}')

    for i, batch in tqdm(enumerate(dataloader)):
        caption = batch['caption']
        path = batch['path']
        images = pipeline(caption, height=img_size[0], width=img_size[1]).images
        
        #save gt image
        if save_gt:
            for p in path:
                name = os.path.basename(p)
                im = Image.open(os.path.join('/mnt/lustrenew/wangxiaodong/data/nuscene', p)).convert('RGB').resize((img_size[1], img_size[0]))
                im.save(os.path.join(save_dir, 'gt', name))
        for (idx, p) in enumerate(path):
            ims = images[idx]
            name = os.path.basename(p)
            ims.save(os.path.join(save_dir, version, name.split('.')[0]+'.jpg'))
    # print('Duplicate each images!')
    print('Save Done!')

if __name__ == '__main__':
    fire.Fire(main)