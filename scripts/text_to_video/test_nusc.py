import argparse
import common
import json
import os
import torch
import timm
from transformers import CLIPTextModel, CLIPTokenizer

import sys
sys.path.append("/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/scripts/text_to_image")

if __name__ == "__main__":
    config_path = "/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/scripts/text_to_video/nusc_256p_video.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    pretrained_model_name_or_path = "/mnt/storage/user/wuzehuan/Downloads/models/stable-diffusion-2-1"
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer",
    )

    train_dataset = common.create_instance_from_config(
            config["training_dataset"])
    # print(training_dataset[0].keys())
    
    batch_size = 2
    view_count = 1
    sequence_length = 8
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
    )
    
    device = "cuda:0"
    layout_encoder = timm.create_model(model_name="convnextv2_base.fcmae", pretrained=True, num_classes=0).to(device)
    
    
    for step, batch in enumerate(train_dataloader):
        _3dbox_images = batch["3dbox_images"].to(device) # [bs, seq, view, c, h, w]
        _3dbox_embeddings = layout_encoder\
                    .forward_features(_3dbox_images.flatten(0, 2))\
                    .flatten(-2).permute(0, 2, 1) #[bs*seq*view, 112, 1024]
        hdmap_images = batch["hdmap_images"].to(device)
        hdmap_embeddings = layout_encoder\
                    .forward_features(hdmap_images.flatten(0, 2))\
                    .flatten(-2).permute(0, 2, 1)
        
        caption = batch["image_description"][0] # the first caption of each batch
        
        inputs = tokenizer(
            caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs_id = inputs['input_ids']
        
        encoder_hidden_states = text_encoder(inputs_id)[0]
        
        print(encoder_hidden_states.shape)
        
        # vae_images
        images = batch["vae_images"].flatten(0, 2) # [bs, seq, view, c, h, w]
    