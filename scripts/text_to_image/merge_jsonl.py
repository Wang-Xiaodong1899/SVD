import json

def merge():

    jsonl_files = ['captions_train_0_32000.jsonl', 'captions_train_32000_64000.jsonl', 'captions_train_64000_96000.jsonl',
                'captions_train_64000_96000.jsonl', 'captions_train_96000_128000.jsonl', 'captions_train_128000_160000.jsonl',
                'captions_train_160000_163367.jsonl']
    output_file = 'captions_train.jsonl'


    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in jsonl_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    outfile.write(line)

    print(f'Merged {len(jsonl_files)} files into {output_file}')

def jsonl2dict():
    merged_dict = {}
    file = 'captions_val.jsonl'
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            utime = entry['utime']
            caption = entry['caption']
            merged_dict[utime] = caption
    
    output_file = 'caption_utime_allframe_val.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_dict, f, ensure_ascii=False, indent=4)

jsonl2dict()