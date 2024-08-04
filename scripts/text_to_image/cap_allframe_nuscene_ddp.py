import torch
import json
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp

from nuscene_image import Allframes

def generate_caption(images, model, processor):
    inputs = processor(images=images, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_caption

def run_ddp(rank, world_size):
    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # 初始化分布式进程组
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # 加载模型和处理器
    processor = AutoProcessor.from_pretrained("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/git-large-coco")
    model = AutoModelForCausalLM.from_pretrained("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/git-large-coco")
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 准备数据集和分布式采样器
    dataset = Allframes(None, None, split='train', img_size=(256, 448), data_root="/mnt/storage/user/wangxiaodong/nuscenes-all")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # 打开文件以写入
    if rank == 0:
        jsonl_file = open('captions.jsonl', 'w')

    for batch in data_loader:
        images, utimes = batch["pixel_values"], batch["utime"]
        captions = generate_caption(images, model, processor)

        # 写入结果到 JSONL 文件
        if rank == 0:
            for utime, caption in zip(utimes, captions):
                json_line = {"utime": utime, "caption": caption}
                jsonl_file.write(json.dumps(json_line) + '\n')

    if rank == 0:
        jsonl_file.close()

def main():
    world_size = torch.cuda.device_count()  # 获取可用GPU数量
    print('world_size: ', world_size)
    mp.spawn(run_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
