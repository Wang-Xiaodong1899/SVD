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
sys.path.append('/home/wxd/video-generation/diffusers/src')
from diffusers import StableDiffusionPipeline

DATAROOT = '/ssd_datasets/wxiaodong/nuscene'
img_size = (192, 384)


class val_dataset(torch.utils.data.Dataset):
    def __init__(self, split='val'):
        super().__init__()
        self.split = split

        # suppose to save original image

        # read from json
        json_path = '/ssd_datasets/wxiaodong/nuscene/nuscene_caption_val.json'
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
        base_model='/ssd_datasets/wxiaodong/ckpt/sd-drive-ep40',
        version = None,
        split='val',
        device='cuda:7',
        save_gt=False,
        save_dir='/ssd_datasets/wxiaodong/nuscene_val',
        size=(256, 512)
):
    os.makedirs(save_dir, exist_ok=True)
    if version==None:
        version = os.path.basename(base_model).split('.')[0]
    print(f'synthsizing image for version {version}')
    if save_gt:
        os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, version), exist_ok=True)

    pipeline = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16, use_safetensors=False, safety_checker=None).to(device)

    print(f'synthsizing image from {split} split')
    dataset = val_dataset(split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, drop_last=False)

    captions = []

    for i, batch in tqdm(enumerate(dataloader)):
        caption = batch['caption']
        path = batch['path']
        images = pipeline(caption, height=size[0], width=size[1]).images
        
        #save gt image
        if save_gt:
            for p in path:
                name = os.path.basename(p)
                im = Image.open(os.path.join('/ssd_datasets/wxiaodong/nuscene', p)).convert('RGB').resize((384, 192))
                im.save(os.path.join(save_dir, 'gt', name))
        for (idx, p) in enumerate(path):
            ims = images[idx]
            name = os.path.basename(p)
            ims.save(os.path.join(save_dir, version, name))
    
    print('Save Done!')

if __name__ == '__main__':
    fire.Fire(main)