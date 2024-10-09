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
sys.path.append('/root/SVD/src')

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from transformers import AutoProcessor, AutoModelForCausalLM

DATAROOT = '/root/autodl-tmp/nuscenes/all/'
img_size = (256, 448)

def load_models(pretrained_model_name_or_path = '/smodels/stable-diffusion-v1-4', unet_path='/root/autodl-fs/smodels/image-keyframes-ep30/unet/', device='cuda:0'):
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

class OVValkeyframes(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer=None, split='val', img_size=(256, 448), data_root="/root/autodl-tmp/nuscenes/all/"):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        self.data_root = data_root

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

        self.splits = create_splits_scenes()

        # training samples
        self.samples_groups = self.group_sample_by_scene(split)
        
        scenes = list(self.samples_groups.keys())
        
        self.all_image_paths = []
        self.search_keys = []
        # re-organize
        # image_paths: []
        # key: [(scene, "idx")]

        self.total_num = 0
        
        for my_scene in scenes:
            image_paths = self.get_paths_from_scene(my_scene)
            self.total_num = self.total_num + len(image_paths)
            indices = [list(range(i, i+8)) for i in range(0, len(image_paths)-8+1, 4)]
            end_index = indices[-1][-1]
            image_paths = image_paths[:end_index]
            self.all_image_paths.extend(image_paths)
            
            selected_index = list(range(len(image_paths)))
            for index in selected_index:
                self.search_keys.append((my_scene, f"{index//4*4}"))

        print('Total samples: %d' % len(self.all_image_paths))

        # search annotations
        json_path = f'/root/SVD/nusc_video_{split}_8_ov-7b_dict.json'
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        # return len(self.all_image_paths)
        return self.total_num # assume 40*150
    
    # indices = [list(range(i, i+8)) for i in range(0, len(paths)-8+1, 4)]
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def __getitem__(self, index):
        try:
            index = index % len(self.all_image_paths)

            image_path = self.all_image_paths[index]
            search_key = self.search_keys[index]
            
            # get dict
            scene_annotation = self.annotations[search_key[0]]
            timestep = search_key[1] if search_key[1] in scene_annotation else str((int(search_key[1])//4-1)*4)
            annotation = scene_annotation[timestep]
            
            caption = ""
            selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
            for attr in selected_attr:
                anno = annotation.get(attr, "")
                if anno == "":
                    continue
                if anno[-1] == ".":
                    anno = anno[:-1] 
                caption = caption + anno + ". "
            
            return {
                'caption': caption,
                'path': image_path,
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


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
        base_model='/smodels/stable-diffusion-v1-4',
        unet_path="/root/autodl-fs/smodels/image-keyframes-ep30/unet/",
        version = None,
        split='val',
        device='cuda:0',
        save_gt=False,
        save_dir='/root/autodl-fs/results',
        size=(256, 448),
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
    dataset = OVValkeyframes(args=None, split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, drop_last=False)

    captions = []

    for i, batch in tqdm(enumerate(dataloader)):
        caption = batch['caption']
        path = batch['path']
        images = pipeline(caption, height=size[0], width=size[1], num_inference_steps=25, guidance_scale=7.5).images
        
        #save gt image
        if save_gt:
            for p in path:
                name = os.path.basename(p)
                im = Image.open(p).convert('RGB').resize((448, 256))
                im.save(os.path.join(save_dir, 'gt', name))
        for (idx, p) in enumerate(path):
            ims = images[idx]
            name = os.path.basename(p)
            ims.save(os.path.join(save_dir, version, name))
    
    print('Save Done!')

if __name__ == '__main__':
    # unet = UNet2DConditionModel.from_pretrained("/root/autodl-fs/smodels/image-ep100/unet")
    # unet.eval()
    # unet.to(torch.float16)
    # unet.save_pretrained("/root/autodl-fs/smodels/image-ep100/unet-fp16")
    fire.Fire(main)