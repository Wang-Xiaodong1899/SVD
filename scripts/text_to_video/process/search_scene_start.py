import json

def load_jsonl(file_path):
    index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            scene = data['scene']
            start = data['start']
            if scene not in index:
                index[scene] = {}
            index[scene][start] = data
    return index

def find_content(index, scene, start):
    if scene in index and start in index[scene]:
        return index[scene][start]
    else:
        return None


def save_index_as_json(index, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=4)

file_path = r'C://Users//wangxiaodong//Desktop//nusc_video_ov_new//nusc_video_val_8_ov-7b_0_150.jsonl'
index = load_jsonl(file_path)


outputfile_path = r'C://Users//wangxiaodong//Desktop//nusc_video_ov_new//nusc_video_val_8_ov-7b_0_150_dict.json'
save_index_as_json(index, outputfile_path)

# scene_to_find = 'scene-0003'
# start_to_find = 0
# result = find_content(index, scene_to_find, start_to_find)

