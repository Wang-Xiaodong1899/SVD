import json
import os
from tqdm import tqdm


DATAROOT = '/ssd_datasets/wxiaodong/nuscene'


def main(
        max_video_len = 8,
        tgt_root = '/home/wxd/data/nuscene/val_150_centerframes',
        split='val'
):
    os.makedirs(tgt_root, exist_ok=True)

    json_path = f'/ssd_datasets/wxiaodong/nuscene/frame_action_{split}.json'
    with open(json_path, 'r') as f:
        frame_action = json.load(f)

    # scenes num
    scenes = list(frame_action.keys())
    print('Total samples: %d' % len(scenes))

    json_path = f'/ssd_datasets/wxiaodong/nuscene/nuscene_caption_utime_{split}.json'
    with open(json_path, 'r') as f:
        caption_utime = json.load(f)
    
    meta_data = []
    
    for sce in tqdm(scenes):
        scene_data = frame_action[sce] #[ [steer], [speed], [path] ]
        inner_frame_len = len(scene_data[0]) # maybe 40

        # center start, fix here
        seek_start = (inner_frame_len - max_video_len)//2

        # add frame
        steers = scene_data[0]
        speeds = scene_data[1]
        paths = scene_data[2]

        seek_steer = steers[seek_start: seek_start+max_video_len]
        seek_speed = speeds[seek_start: seek_start+max_video_len]
        seek_path = paths[seek_start: seek_start+max_video_len]

        first_image_path = seek_path[0]
        utime = first_image_path.split('__')[-1].split('.')[0]
        first_image_caption = caption_utime[utime]

        meta_data.append(
            {
                'path': first_image_path,
                'caption': first_image_caption,
            }
        )
        # copy file to target dir
        source_file = os.path.join(DATAROOT, first_image_path)
        os.system(f'cp {source_file} {tgt_root}')

    # save metadata
    with open(os.path.join(tgt_root, 'meta.json'), 'w') as f:
        json.dump(meta_data, f)
    
    print('saved meta data')

if __name__ == '__main__':
    main()