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

DATAROOT = '/ssd_datasets/wxiaodong/nuscene'


class keyframes(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='train', img_size=(256, 512)):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split

        # self.nusc = NuScenes(version='v1.0', dataroot=DATAROOT, verbose=True)

        # self.splits = create_splits_scenes()

        # training samples
        # self.samples = self.get_samples(split)

        # image transform
        self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.), ratio=(1., 1.78)),
                # transforms.RandomHorizontalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # text tokenizer

        # read from json
        json_path = '/ssd_datasets/wxiaodong/nuscene/nuscene_caption_train.json'
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        print('Total samples: %d' % len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def __getitem__(self, index):
        try:
            # my_sample = self.samples[index]

            # only front camera
            # file_path = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])['filename']

            my_sample = self.samples[index]
            file_path = my_sample['path']
            caption = my_sample['caption']

            image_path = os.path.join(DATAROOT, file_path)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            # suppose we have caption for each sample
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