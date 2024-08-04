from transformers import CLIPTextModel, CLIPTokenizer
from nuscene_action_dwm import ActionAllFrames
from action2video_dwm_config import LlamaConfig

pretrained_model_name_or_path = '/mnt/storage/user/wangxiaodong/DWM_work_dir/lidar_maskgit_debug/smodels/image-ep100'

config = LlamaConfig()

tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer", revision=None
)

dataset = ActionAllFrames(config, tokenizer, split='train', img_size=(192, 384), data_root="/mnt/storage/user/wangxiaodong/nuscenes-all")
print(len(dataset))

print(dataset[0])