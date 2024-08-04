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

DATAROOT = '/mnt/storage/user/wangxiaodong/nuscenes'

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

#  resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


# default image size 256x512

class Actionframes(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='train', history_len = 8, max_video_len = 8, img_size=(192, 384)):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        self.max_video_len = max_video_len
        self.action_scale = 1
        clip_size = (224, 224)

        # image transform
        self.transform = transforms.Compose([
                transforms.ToPILImage('RGB'),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        
        self.clip_transform = transforms.Normalize(
                transformers.image_utils.OPENAI_CLIP_MEAN,
                transformers.image_utils.OPENAI_CLIP_STD)
        self.clip_resize = transforms.Resize(clip_size)

        # read from json
        json_path = f'/mnt/storage/user/wangxiaodong/nuscenes/scene_action_file_{split}.json'
        with open(json_path, 'r') as f:
            self.scene_action = json.load(f)
        
        # scenes num
        print('Total Scene: %d' % len(self.scene_action))

        # utime-caption
        json_path = f'/mnt/storage/user/wangxiaodong/nuscenes/nuscene_caption_utime_{split}.json'
        with open(json_path, 'r') as f:
            self.caption_utime = json.load(f)
    
    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def __len__(self):
        return len(self.scene_action)
    
    def __getitem__(self, index):
        try:
            item = self.scene_action[index]
            my_scene = item['scene']
            files = item['files']
            angles = item['angles']
            speeds = item['speeds']

            mini_len = min(len(angles), len(speeds), len(files))
            files = files[:mini_len]
            angles = angles[:mini_len]
            speeds = speeds[:mini_len]

            if self.split == 'train':
                # start from history_len
                seek_start = random.randint(0, mini_len - self.args.max_seq_len)
                seek_angle = angles[seek_start: seek_start+self.args.max_video_len+self.args.history_len]
                seek_speed = speeds[seek_start: seek_start+self.args.max_video_len+self.args.history_len]
                seek_path = files[seek_start: seek_start+self.args.max_video_len+self.args.history_len]
            else:
                seek_start = 0
                seek_angle = angles[seek_start: self.args.valid_max_video_len]
                seek_speed = speeds[seek_start: self.args.valid_max_video_len]
                seek_path = files[seek_start: self.args.valid_max_video_len]

            seek_angle_ms = torch.tensor(seek_angle) # default [-9, 9]
            seek_speed_ms = torch.tensor(seek_speed) / 3.6

            # NOTE we skip 8 action
            src_action_len = len(seek_angle_ms) - 1 # leave for bos
            tgt_action_len = len(seek_angle_ms)
            attention_mask = [1] * (1 + src_action_len) + [0] * (self.args.max_seq_len - len(seek_angle_ms) - 1) # start with bos embed
            loss_mask = [0] * self.args.history_len + [1] * (tgt_action_len - self.args.history_len) + [0] * (self.args.max_seq_len - len(seek_angle_ms))

            utimes = [p.split('__')[-1].split('.')[0] for p in seek_path]
            captions = [self.caption_utime[ut] for ut in utimes]

            first_image_path = seek_path[0]
            utime = first_image_path.split('__')[-1].split('.')[0]
            first_image_caption = self.caption_utime[utime]

            frame_paths = [os.path.join(DATAROOT, file_path) for file_path in seek_path]
            video = np.stack([image2arr(fn) for fn in frame_paths])

            img = image2arr(frame_paths[0]) # first image

            state = torch.get_rng_state()
            video = torch.stack([self.augmentation(v, self.transform, state) for v in video], dim=0)
            # if video.shape[0] < self.max_video_len:
            #     video = torch.cat((video, repeat(video[-1], 'c h w -> n c h w',
            #                                      n=self.max_video_len - video.shape[0])), dim=0)
            # imgs = video[-(self.args.max_video_len+self.args.video_offset)+1:] # NOTE need last frames
            imgs = video[: self.args.history_len]

            # NOTE take correct caption
            # print(f'caption length: {len(captions)}')
            # print(f'max_video: {}')
            # print(f'idx: {-(self.args.max_video_len+self.args.video_offset)}')
            # first_image_caption = captions[-(self.args.max_video_len+self.args.video_offset)+1] 
            first_image_caption = captions[0]

            clip_video = imgs # n c h w
            clip_video = _resize_with_antialiasing(clip_video, (224, 224))
            clip_video = (clip_video + 1.0) / 2.0 # -> (0, 1)
            clip_video = self.clip_transform(clip_video)

            # suppose use caption for first sample
            inputs = self.tokenizer(
                first_image_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            inputs_id = inputs['input_ids'][0] # no question

            return {
                'input_ids': inputs_id,
                'label_imgs': imgs,
                'pil': img,
                'steer': seek_angle_ms,
                'speed': seek_speed_ms,
                'clip_imgs': clip_video,
                'caption': first_image_caption,
                'name': os.path.basename(first_image_path),
                'scene': my_scene,
                'attention_mask': torch.Tensor(attention_mask).to(torch.float),
                'loss_mask': torch.Tensor(loss_mask).to(torch.float),
                'captions': captions
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))



class ActionAllFrames(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='train', history_len = 8, max_video_len = 8, img_size=(192, 384), data_root=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        self.max_video_len = max_video_len
        self.action_scale = 1
        self.data_root = data_root if data_root else DATAROOT
        
        clip_size = (224, 224)

        # image transform
        self.transform = transforms.Compose([
                transforms.ToPILImage('RGB'),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        
        self.clip_transform = transforms.Normalize(
                transformers.image_utils.OPENAI_CLIP_MEAN,
                transformers.image_utils.OPENAI_CLIP_STD)
        self.clip_resize = transforms.Resize(clip_size)

        # read from json
        json_path = f'/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/scripts/action_to_video/scene_action_file_allframes_{split}.json'
        with open(json_path, 'r') as f:
            self.scene_action = json.load(f)
        
        # scenes num
        print('Total Scene: %d' % len(self.scene_action))

        # utime-caption
        json_path = f'/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/scripts/text_to_image/caption_utime_allframe_{split}.json'
        with open(json_path, 'r') as f:
            self.caption_utime = json.load(f)
    
    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def __len__(self):
        return len(self.scene_action)
    
    def __getitem__(self, index):
        try:
            item = self.scene_action[index]
            my_scene = item['scene']
            files = item['files']
            angles = item['angles']
            speeds = item['speeds']

            mini_len = min(len(angles), len(speeds), len(files))
            files = files[:mini_len]
            angles = angles[:mini_len]
            speeds = speeds[:mini_len]

            if self.split == 'train':
                # start from history_len
                seek_start = random.randint(0, mini_len - self.args.max_seq_len)
                seek_angle = angles[seek_start: seek_start+self.args.max_video_len+self.args.history_len]
                seek_speed = speeds[seek_start: seek_start+self.args.max_video_len+self.args.history_len]
                seek_path = files[seek_start: seek_start+self.args.max_video_len+self.args.history_len]
            else:
                seek_start = 0
                seek_angle = angles[seek_start: self.args.valid_max_video_len]
                seek_speed = speeds[seek_start: self.args.valid_max_video_len]
                seek_path = files[seek_start: self.args.valid_max_video_len]

            seek_angle_ms = torch.tensor(seek_angle) # default [-9, 9]
            seek_speed_ms = torch.tensor(seek_speed) / 3.6

            # NOTE we skip 8 action
            src_action_len = len(seek_angle_ms) - 1 # leave for bos
            tgt_action_len = len(seek_angle_ms)
            attention_mask = [1] * (1 + src_action_len) + [0] * (self.args.max_seq_len - len(seek_angle_ms) - 1) # start with bos embed
            loss_mask = [0] * self.args.history_len + [1] * (tgt_action_len - self.args.history_len) + [0] * (self.args.max_seq_len - len(seek_angle_ms))

            utimes = [p.split('__')[-1].split('.')[0] for p in seek_path]
            captions = [self.caption_utime[ut] for ut in utimes]

            first_image_path = seek_path[0]
            utime = first_image_path.split('__')[-1].split('.')[0]
            first_image_caption = self.caption_utime[utime]

            frame_paths = [os.path.join(self.data_root, file_path) for file_path in seek_path]
            video = np.stack([image2arr(fn) for fn in frame_paths])

            img = image2arr(frame_paths[0]) # first image

            state = torch.get_rng_state()
            video = torch.stack([self.augmentation(v, self.transform, state) for v in video], dim=0)
            # if video.shape[0] < self.max_video_len:
            #     video = torch.cat((video, repeat(video[-1], 'c h w -> n c h w',
            #                                      n=self.max_video_len - video.shape[0])), dim=0)
            # imgs = video[-(self.args.max_video_len+self.args.video_offset)+1:] # NOTE need last frames
            imgs = video[: self.args.history_len]

            # NOTE take correct caption
            # print(f'caption length: {len(captions)}')
            # print(f'max_video: {}')
            # print(f'idx: {-(self.args.max_video_len+self.args.video_offset)}')
            # first_image_caption = captions[-(self.args.max_video_len+self.args.video_offset)+1] 
            first_image_caption = captions[0]

            clip_video = imgs # n c h w
            clip_video = _resize_with_antialiasing(clip_video, (224, 224))
            clip_video = (clip_video + 1.0) / 2.0 # -> (0, 1)
            clip_video = self.clip_transform(clip_video)

            # suppose use caption for first sample
            inputs = self.tokenizer(
                first_image_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            inputs_id = inputs['input_ids'][0] # no question

            return {
                'input_ids': inputs_id,
                'label_imgs': imgs,
                'pil': img,
                'steer': seek_angle_ms,
                'speed': seek_speed_ms,
                'clip_imgs': clip_video,
                'caption': first_image_caption,
                'name': os.path.basename(first_image_path),
                'scene': my_scene,
                'attention_mask': torch.Tensor(attention_mask).to(torch.float),
                'loss_mask': torch.Tensor(loss_mask).to(torch.float),
                'captions': captions
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))
