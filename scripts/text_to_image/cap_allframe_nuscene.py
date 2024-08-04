import torch
import json
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from nuscene_image import Allframes

import fire

def generate_caption(images, model, processor, device):
    pil_list = []
    for path in images:
        pil_list.append(Image.open(path).convert('RGB'))
    
    inputs = processor(images=pil_list, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_caption

def run(device='cuda:0', split='val', start_idx=0, end_idx=32000):

    # 加载模型和处理器
    processor = AutoProcessor.from_pretrained("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/git-large-coco")
    model = AutoModelForCausalLM.from_pretrained("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/git-large-coco")
    model.to(device)
    model = model.to(torch.float16)

    # 准备数据集和分布式采样器
    dataset = Allframes(None, None, split=split, img_size=(256, 448), data_root="/mnt/storage/user/wangxiaodong/nuscenes-all")
    
    subset = Subset(dataset, range(start_idx, end_idx))
    
    dataloader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

    jsonl_file = open(f'captions_{split}_{start_idx}_{end_idx}.jsonl', 'w')

    for batch in tqdm(dataloader):
        images, utimes = batch["image_path"], batch["utime"]
        captions = generate_caption(images, model, processor, device)

        for utime, caption in zip(utimes, captions):
            json_line = {"utime": str(utime.item()), "caption": caption}
            jsonl_file.write(json.dumps(json_line) + '\n')
        
        jsonl_file.flush()

    jsonl_file.close()


if __name__ == '__main__':
    fire.Fire(run)
