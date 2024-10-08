import torch
import torch.nn as nn
import os
import sys
sys.path.append('/workspace/wxd/SVD/src')
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPImageProcessor
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler, UNet2DConditionModel

# prepare components
device = 'cuda:0'

# prepare pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
                "/volsparse1/wxd/smodels/image-keyframes-ep30", safety_checker=None
            )
pipeline = pipeline.to(device).to(torch.float16)

import pdb; pdb.set_trace()

count = 0

while True:
    prompt = input()
    image = pipeline(prompt=prompt,height=256, width=448).images[0]
    image.save(f'ov-sd-gen_{count}.jpg')
    count = count + 1