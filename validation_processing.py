import os
import shutil

dir_list = os.listdir("./tiny-imagenet-200/train")
for name in dir_list:
    os.mkdir(os.path.join("./tiny-imagenet-200/val", name))
with open("./tiny-imagenet-200/val_copy/val_annotations.txt") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        label_val = line.split("\t")[1]
        src = "./tiny-imagenet-200/val_copy/images/val_" + str(i) + ".JPEG"
        dst = "./tiny-imagenet-200/val/" + label_val + "/val_" + str(i) +".JPEG"
        shutil.copy2(src, dst)