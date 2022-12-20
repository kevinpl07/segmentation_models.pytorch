import os
import torch
import shutil
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import random
from torchvision import transforms
import json
import cv2

class VideoMattingDataset(torch.utils.data.Dataset):
    def __init__(self, root, backgrounds, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.filenames = self._read_split() # read train/valid/test splits
        self.bgs = glob.glob(os.path.join(backgrounds, "*.png")) + glob.glob(os.path.join(backgrounds, "*.jpg"))
        # self.bg_masks = [glob.glob(os.path.join(backgrounds, "*.png"))]

        # label_file = backgrounds + "labels.json"

        # with open(label_file, 'r', encoding="utf-8") as f:
        #     labels = json.load(f)

        # for img_name, label_dict in labels.items():
        #     is_background = False
        #     for _, region_dict in label_dict["regions"].items():
        #         if region_dict["region_attributes"]["label"] == "":
        #             is_background = True
        #             break

        #     if is_background:
        #         img_shape_orig = cv2.imread(backgrounds + img_name).shape
        #         mask = np.zeros(img_shape_orig[:2], dtype=np.uint8)

        #         # Mask out foreground objects
        #         for region_key, region_dict in label_dict["regions"].items():
        #             if region_dict['region_attributes']["label"] != "Background":
        #                 polygon = region_dict['shape_attributes']
        #                 x_points = np.asarray(polygon['all_points_x'])
        #                 y_points = np.asarray(polygon['all_points_y'])
        #                 points = np.asarray(list(zip(x_points, y_points)))
        #                 cv2.fillPoly(mask, np.asarray([points], dtype=np.int32), 1)

        #         # cutout background parts INSIDE forground 
        #         # for region_key, region_dict in label_dict["regions"].items():
        #         #     if region_dict['region_attributes']["label"] == "Background":
        #         #         polygon = region_dict['shape_attributes']
        #         #         x_points = np.asarray(polygon['all_points_x'], dtype=np.uint8)
        #         #         y_points = np.asarray(polygon['all_points_y'], dtype=np.uint8)
        #         #         points = np.vstack((x_points, y_points)).T
        #         #         points = np.sort(points)
        #         #         cv2.fillConvexPoly(mask, np.asarray(points, dtype=np.int32), 0)

            
        #         self.bgs.append(backgrounds + img_name)
        #         self.bg_masks.append(mask)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        pair = self.filenames[idx]
        image_path = pair[0]
        mask_path = pair[1]
        true_fgr = Image.open(image_path)
        width, height = true_fgr.size
        new_height = 650
        new_width = int(650 * (width / height))

        true_fgr = true_fgr.resize((new_width, new_height))
        true_pha = Image.open(mask_path).convert('L').resize((new_width, new_height))

        max_offset = int(new_width - 786)

        offset = random.randrange(max_offset)

        true_fgr = true_fgr.crop((offset, 0, (offset+786), 650))
        true_pha = true_pha.crop((offset, 0, (offset+786), 650))


        bg_idx = random.randrange(len(self.bgs))

        true_bgr = Image.open(self.bgs[bg_idx]).convert("RGB").resize((786, 650))
        # true_pha = np.array(true_pha)

        # print(true_pha.max())
        # print(true_pha.min())
        # print(true_pha)
        # print(true_pha.dtype)


        # true_bgr_pha = cv2.resize(self.bg_masks[bg_idx], (true_pha.size[0], true_pha.size[1]))
        # true_pha[true_pha <= 1] = 0
        # true_pha[true_pha > 1] = 1
        # true_pha = Image.fromarray(np.asarray(cv2.bitwise_or(true_pha, bg_mask) * 255, dtype=np.int8))

        sample = dict(true_fgr=true_fgr, true_pha=true_pha, true_bgr=true_bgr)#, true_bgr_pha=true_bgr_pha)

        if self.transform is not None:
            sample = {k: self.transform(v) for k, v in sample.items()}

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask
    

    def _read_split(self):
        folder_name = "test" if self.mode == "test" else "train"
        img_fnames = sorted(glob.glob(os.path.join(self.root, folder_name, "fgr", "*", "*.jpg")))
        mask_fnames = [img_name.replace("fgr", "pha") for img_name in img_fnames]
        if self.mode == "train":  # 90% for train
            img_fnames = [x for i, x in enumerate(img_fnames) if i % 10 != 0]
            mask_fnames = [x for i, x in enumerate(mask_fnames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            img_fnames = [x for i, x in enumerate(img_fnames) if i % 10 == 0]
            mask_fnames = [x for i, x in enumerate(mask_fnames) if i % 10 == 0]
        return list(zip(img_fnames, mask_fnames))

    # @staticmethod
    # def download(root):

    #     # load images
    #     filepath = os.path.join(root, "images.tar.gz")
    #     download_url(
    #         url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    #         filepath=filepath,
    #     )
    #     extract_archive(filepath)

    #     # load annotations
    #     filepath = os.path.join(root, "annotations.tar.gz")
    #     download_url(
    #         url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
    #         filepath=filepath,
    #     )
    #     extract_archive(filepath)
