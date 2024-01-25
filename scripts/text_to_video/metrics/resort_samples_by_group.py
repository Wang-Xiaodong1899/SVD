import os
import json
from common import json2data, data2json

def main():
    meta = json2data('/home/wxd/data/nuscene/samples_group_val.json')

    resort_meta = []
    for item in meta:
        scene = item['scene']
        samples = item['samples']

        samples = [os.path.basename(p) for p in samples]
        sorted_samples = sorted(samples, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        resort_meta.append(
            {
                'scene': scene,
                'samples': sorted_samples
            }
        )
    data2json(resort_meta, '/home/wxd/data/nuscene/samples_group_sort_val.json')

if __name__ == "__main__":
    main()