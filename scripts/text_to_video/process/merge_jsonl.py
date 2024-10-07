import os
import json

# Define the input folder path and output file path
input_folder = r'C://Users//wangxiaodong//Desktop//nusc_video_ov'
output_file = 'nusc_video_val_8_ov-7b_0_150.jsonl'

# Create the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate through each file in the folder
    filenames = list(os.listdir(input_folder))
    filenames.sort()
    print(filenames)
    for filename in filenames:
        if filename.endswith('.jsonl') and filename.startswith('nusc_video_val'):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

print("Merge complete!")
