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

DATAROOT = '/mnt/lustrenew/wangxiaodong/data/nuscene'

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

class Videoframes(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='train', max_video_len = 8, img_size=(256,512)):
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
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.), ratio=(1., 1.777778)),
                # transforms.RandomHorizontalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        
        self.clip_transform = transforms.Normalize(
                transformers.image_utils.OPENAI_CLIP_MEAN,
                transformers.image_utils.OPENAI_CLIP_STD)
        self.clip_resize = transforms.Resize(clip_size)

        # read from json
        json_path = '/mnt/lustrenew/wangxiaodong/data/nuscene/frame_action_train.json'
        with open(json_path, 'r') as f:
            self.frame_action = json.load(f)
        
        # scenes num
        self.scenes = list(self.frame_action.keys())
        print('Total samples: %d' % len(self.scenes))

        # utime-caption
        json_path = '/mnt/lustrenew/wangxiaodong/data/nuscene/nuscene_caption_utime_train.json'
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
                'steer': seek_steer,
                'speed': seek_speed,
                'clip_imgs': clip_video
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

class VideoAllframes(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, split='train', max_video_len = 8, img_size=(256, 448), data_root="/mnt/storage/user/wangxiaodong/nuscenes-all"):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        self.max_video_len = max_video_len
        self.action_scale = 1
        clip_size = (224, 224)
        
        self.data_root = data_root if data_root else DATAROOT
        
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

        self.splits = create_splits_scenes()

        # training samples
        self.samples = self.get_samples(split)

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

        # utime-caption
        json_path = f'/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/scripts/text_to_image/caption_utime_allframe_{split}.json'
        with open(json_path, 'r') as f:
            self.caption_utime = json.load(f)
        
        # scenes num
        print('Total samples: %d' % len(self.samples))
    
    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
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
        
        samples_group_by_scene = []
        for scene in selected_scenes_meta:
            samples_group_by_scene.append(
                {
                'scene': scene['name'],
                'video': self.get_all_frames_from_scene(scene)
                }
            )
        
        return samples_group_by_scene
    
    
    def __getitem__(self, index):
        try:
            my_scene = self.samples[index]
            
            my_sample_list = my_scene["video"]
            inner_frame_len = len(my_sample_list)
            
            print(f'inner_frame_len: {inner_frame_len}')

            # always skip 1 frame
            seek_start = random.randint(0, inner_frame_len - self.max_video_len)
            seek_samples = my_sample_list[seek_start: seek_start+self.max_video_len]

            # for first sample
            my_sample = seek_samples[0]
            sample_token = my_sample["sample_token"]
            scene_token = self.nusc.get('sample', sample_token)["scene_token"]
            desc = self.nusc.get('scene', scene_token)["description"]
            utime = my_sample['timestamp']
            first_image_caption = self.caption_utime[str(utime)]
            
            # clip all frames
            frame_paths = []
            for my_sample in seek_samples:
                file_path = my_sample["filename"]
                image_path = os.path.join(self.data_root, file_path)
                frame_paths.append(image_path)
            video = np.stack([image2arr(fn) for fn in frame_paths])

            state = torch.get_rng_state()
            video = torch.stack([self.augmentation(v, self.transform, state) for v in video], dim=0)
            if video.shape[0] < self.max_video_len:
                video = torch.cat((video, repeat(video[-1], 'c h w -> n c h w',
                                                 n=self.max_video_len - video.shape[0])), dim=0)
            imgs = video[:self.max_video_len]

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
                'clip_imgs': clip_video
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))
