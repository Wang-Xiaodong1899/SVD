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
from einops import repeat


from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

DATAROOT = '/ssd_datasets/wxiaodong/nuscene'

def image2pil(filename):
    return Image.open(filename)

def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)

def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr


# default image size 256x512

class Videoframes(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='train', max_video_len = 8, img_size=(256,512)):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        self.max_video_len = max_video_len
        self.action_scale = 1

        # image transform
        self.transform = transforms.Compose([
                transforms.ToPILImage('RGB'),
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.), ratio=(1., 1.78)),
                # transforms.RandomHorizontalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # text tokenizer

        # read from json
        json_path = '/ssd_datasets/wxiaodong/nuscene/frame_action_train.json'
        with open(json_path, 'r') as f:
            self.frame_action = json.load(f)
        
        # scenes num
        self.scenes = list(self.frame_action.keys())
        print('Total samples: %d' % len(self.scenes))

        # utime-caption
        json_path = '/ssd_datasets/wxiaodong/nuscene/nuscene_caption_utime_train.json'
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

            # alwys skip 1 frame
            seek_start = random.randint(0, inner_frame_len - self.max_video_len)

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

            state = torch.get_rng_state()
            video = torch.stack([self.augmentation(v, self.transform, state) for v in video], dim=0)
            if video.shape[0] < self.max_video_len:
                video = torch.cat((video, repeat(video[-1], 'c h w -> n c h w',
                                                 n=self.max_video_len - video.shape[0])), dim=0)
            imgs = video[:self.max_video_len]

            # suppose use caption for first sample
            inputs = self.tokenizer(
                first_image_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            inputs_id = inputs['input_ids'][0] # no question

            return {
                'input_ids': inputs_id,
                'label_imgs': imgs,
                'steer': seek_steer,
                'speed': seek_speed
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))