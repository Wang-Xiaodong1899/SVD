
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import fire

from common import image2arr, pil2arr, mp4toarr, image2pil, json2data

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception import InceptionV3


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        # path -> frame path
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

class VideoPathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, frames=38):
        self.files = files
        self.transforms = transforms
        self.frames = frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        # path -> video path

        video_arr = mp4toarr(path)
        video = video_arr[:self.frames]
        images = []
        for frame in video:
            img = Image.fromarray(frame).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            images.append(img)
        images = torch.stack(images, 0)
        return images


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def main(
    tgt_dir='/mnt/lustrenew/wangxiaodong/data/nuscene/FVD-first-15/video-v11-ep200-s196',
    version = None,
    num_frames=8,
    eval_frames=8,
    batch_size=16, 
    device='cuda', 
    dims=2048, 
    num_workers=4
):

    if version is None:
        version = os.path.basename(tgt_dir)+f'_F{num_frames}'
    
    print(f'eval videos at {tgt_dir} with {eval_frames} frames')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    print('loaded iception model')

    # read real video
    meta_data = json2data('/mnt/lustrenew/wangxiaodong/data/nuscene/samples_group_sort_val.json')
    files_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/val_group'

    syn_videos_paths = []
    # video_files = os.listdir(tgt_dir)
    for item in tqdm(meta_data):
        sce = item['scene']
        files = os.listdir(os.path.join(tgt_dir, sce))
        for file in files:
            if file.split('.')[-1] == 'mp4':
                syn_videos_paths.append(os.path.join(tgt_dir, sce, file))
    print(f'syn length {len(syn_videos_paths)}')

    real_frames_paths = []
    for item in tqdm(meta_data):
        sce = item['scene']
        frames_info = item['samples']
        frames = []
        for im_path in frames_info[:eval_frames]:
            real_frames_paths.append(os.path.join(files_dir, sce, os.path.basename(im_path)))

    print(f'GT length {len(real_frames_paths)}')
    
    # video dataset
    video_dataset = VideoPathDataset(syn_videos_paths, transforms=TF.ToTensor(), frames=eval_frames)
    video_dataloader = torch.utils.data.DataLoader(video_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    frame_dataset = ImagePathDataset(real_frames_paths, transforms=TF.ToTensor())
    frame_dataloader = torch.utils.data.DataLoader(frame_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    # video
    pred_arr = np.empty((len(syn_videos_paths)*eval_frames, dims))
    print(f'pred arr shape: {pred_arr.shape}')

    start_idx = 0

    for batch in tqdm(video_dataloader):
        batch = batch.to(device)
        batch = batch.flatten(0,1)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]
    
    mu_pred = np.mean(pred_arr, axis=0)
    sigma_pred = np.cov(pred_arr, rowvar=False)

    # frames
    gt_arr = np.empty((len(real_frames_paths), dims))
    print(f'gt arr shape: {pred_arr.shape}')

    start_idx = 0

    for batch in tqdm(frame_dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        gt_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    mu_gt = np.mean(gt_arr, axis=0)
    sigma_gt = np.cov(gt_arr, rowvar=False)

    fid_value = calculate_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)

    print('FID: ', fid_value)


if __name__ == '__main__':
    fire.Fire(main)
