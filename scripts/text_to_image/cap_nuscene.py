#TODO load nuscene data and load model to inference the image caption
import os
import torch
import torchvision
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM
from PIL import Image
import json
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# Use GIT-Large to caption keyframes

git_processor_large = AutoProcessor.from_pretrained("/mnt/lustrenew/wangxiaodong/models/git-large-coco")
git_model_large = AutoModelForCausalLM.from_pretrained("/mnt/lustrenew/wangxiaodong/models/git-large-coco")

print('model loaded!')
device = 'cuda:0'

git_model_large.to(device)


def generate_caption(image):
    pil_list = []
    for path in image:
        pil_list.append(Image.open(path).convert('RGB'))
    inputs = git_processor_large(images=pil_list, return_tensors="pt").to(device)
    
    generated_ids = git_model_large.generate(pixel_values=inputs.pixel_values, max_length=50)

    generated_caption = git_processor_large.batch_decode(generated_ids, skip_special_tokens=True)
   
    return generated_caption

DATAROOT = '/mnt/lustrenew/wangxiaodong/data/nuscene'
# image size 1600x900

class frameDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.split = split

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)

        self.splits = create_splits_scenes()

        # training samples
        self.samples = self.get_samples(split)
        print('Total samples: %d' % len(self.samples))

    def __len__(self):
        return len(self.samples)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def __getitem__(self, index):
        try:
            my_sample = self.samples[index]
            # only front camera
            file_path = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])['filename']
            image_path = os.path.join(DATAROOT, file_path)
            # image = Image.open(image_path).convert('RGB')

            return {
                'path': file_path,
                'pixel_values': image_path,
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            # return self.__getitem__(np.random.randint(0, self.__len__() - 1))

def main(
        split='train',
):
    print(f'captioning {split} split')
    dataset = frameDataset(split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

    captions = []

    for i, batch in tqdm(enumerate(dataloader)):
        # print(batch['path'])
        caption = generate_caption(batch['pixel_values'])
        # print(caption)
        for idx, cap in enumerate(caption):
            captions.append(
                {
                'path': batch['path'][idx],
                'caption': cap
                }
            )
    
    # save for a json 
    with open(f'/mnt/lustrenew/wangxiaodong/data/nuscene/nuscene_caption_{split}.json', 'w') as f:
        json.dump(captions, f)
    print('Save Done!')

if __name__ == '__main__':
    main(split='train')