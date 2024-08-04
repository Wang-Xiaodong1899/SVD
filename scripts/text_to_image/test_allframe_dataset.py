from nuscene_image import Allframes


dataset = Allframes(None, None, split='train', img_size=(256, 448), data_root="/mnt/storage/user/wangxiaodong/nuscenes-all")
print(len(dataset))

print(dataset[0])