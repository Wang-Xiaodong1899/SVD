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


from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

DATAROOT = '/mnt/storage/user/wangxiaodong/nuscenes'


class Allframes(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='train', img_size=(256, 448), data_root="/mnt/storage/user/wangxiaodong/nuscenes-all"):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        self.data_root = data_root if data_root else DATAROOT

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

        self.splits = create_splits_scenes()

        # training samples
        self.samples = self.get_samples(split)

        # image transform
        self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # read from json
        # json_path = '/mnt/lustrenew/wangxiaodong/data/nuscene/nuscene_caption_train.json'
        # with open(json_path, 'r') as f:
        #     self.samples = json.load(f)
        print('Total samples: %d' % len(self.samples))

        # utime-caption
        
        json_path = f'/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/scripts/text_to_image/caption_utime_allframe_{split}.json'
        with open(json_path, 'r') as f:
            self.caption_utime = json.load(f)
    
    def __len__(self):
        return len(self.samples)
    
    def get_all_frames_from_scene(self, scene):
        # get all frames (keyframes, sweeps)
        first_sample_token = scene['first_sample_token']
        my_sample = self.nusc.get('sample', first_sample_token)
        sensor = "CAM_FRONT"
        cam_front_data = self.nusc.get('sample_data', my_sample['data'][sensor]) # first frame sensor token
        # frames = 0
        all_frames_dict = [] # len() -> frame number
        while True:
            all_frames_dict.append(cam_front_data)
            # filename = cam_front_data['filename']  # current-frame filename
            next_sample_data_token = cam_front_data['next']  # next-frame sensor token
            if not next_sample_data_token: # ''
                break
            cam_front_data = self.nusc.get('sample_data', next_sample_data_token)
            # frames += 1
        
        return all_frames_dict
    
    
    def get_samples(self, split='train'):
        selected_scenes = self.splits[split] # all scenes
        all_scenes = self.nusc.scene
        selected_scenes_meta = []
        for sce in all_scenes:
            if sce['name'] in selected_scenes:
                selected_scenes_meta.append(sce)
        
        # samples_group_by_scene = []
        all_samples_data = []
        for scene in selected_scenes_meta:
            all_samples_data.extend(
                self.get_all_frames_from_scene(scene)
            )
            # samples_group_by_scene.append(
            #     {
            #         'scene': scene,
            #         'data': self.get_all_frames_from_scene(scene)
            #     }
            # )
        
        return all_samples_data
    
    def __getitem__(self, index):
        try:
            my_sample = self.samples[index]
            
            sample_token = my_sample["sample_token"]
            scene_token = self.nusc.get('sample', sample_token)["scene_token"]
            desc = self.nusc.get('scene', scene_token)["description"]
            
            utime = my_sample['timestamp']

            caption = self.caption_utime[str(utime)]

            file_path = my_sample["filename"]
            image_path = os.path.join(self.data_root, file_path)
            image = Image.open(image_path).convert('RGB')
            
            # pil to tensor
            image = self.transform(image)

            if self.tokenizer is not None:
                inputs = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                )
                inputs_id = inputs['input_ids'][0]
            else:
                inputs_id = ""

            return {
                'input_ids': inputs_id,
                'pixel_values': image,
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))




class keyframes(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='train', img_size=(192, 384)):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)

        self.splits = create_splits_scenes()

        # training samples
        self.samples = self.get_samples(split)

        # image transform
        self.transform = transforms.Compose([
                transforms.Resize(img_size),
                # transforms.RandomResizedCrop(img_size, scale=(0.5, 1.), ratio=(1., 1.78)),
                transforms.RandomHorizontalFlip(p=0.1),
                # transforms.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # read from json
        # json_path = '/mnt/lustrenew/wangxiaodong/data/nuscene/nuscene_caption_train.json'
        # with open(json_path, 'r') as f:
        #     self.samples = json.load(f)
        print('Total samples: %d' % len(self.samples))

        # utime-caption
        json_path = f'/mnt/storage/user/wangxiaodong/nuscenes/nuscene_caption_utime_{split}.json'
        with open(json_path, 'r') as f:
            self.caption_utime = json.load(f)
    
    def __len__(self):
        return len(self.samples)
    
    def get_samples(self, split='train'):
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

            sample_token = sample_data["sample_token"]
            scene_token = self.nusc.get('sample', sample_token)["scene_token"]
            desc = self.nusc.get('scene', scene_token)["description"]


            utime = file_path.split('__')[-1].split('.')[0]

            caption = self.caption_utime[utime]

            image_path = os.path.join(DATAROOT, file_path)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            # suppose we have caption for each sample
            # NOTE if use coarse caption

            inputs = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            inputs_id = inputs['input_ids'][0]

            return {
                'input_ids': inputs_id,
                'pixel_values': image,
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))