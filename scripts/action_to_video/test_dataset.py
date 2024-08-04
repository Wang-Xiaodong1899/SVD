from nuscene_video import VideoAllframes

from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path = "/mnt/storage/user/wuzehuan/Downloads/models/stable-diffusion-2-1"
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16"
)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer",
)


dataset = VideoAllframes(None, tokenizer, split='train', img_size=(256, 448), data_root="/mnt/storage/user/wangxiaodong/nuscenes-all")
print(len(dataset))

print(dataset[0])