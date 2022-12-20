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

class StreamfogDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", augment_with_other_bgs=False):

        assert mode in {"train", "valid"}

        self.root = root
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.augment_with_other_bgs = augment_with_other_bgs

        self.filenames = glob.glob(os.path.join(root, mode, "*.png"))
        if self.augment_with_other_bgs:
            self.bgs = glob.glob(os.path.join(root, 'bgs', "*.jpg"))
        self.bg_masks = []

        label_file = root + "labels.json"

        with open(label_file, 'r', encoding="utf-8") as f:
            labels = json.load(f)


        for img_name in self.filenames:
            label_dict = labels[os.path.basename(img_name)]

            img_shape_orig = cv2.imread(img_name).shape
            mask = np.zeros(img_shape_orig[:2], dtype=np.uint8)

            # Mask out foreground objects
            for _, region_dict in label_dict["regions"].items():
                if region_dict['region_attributes']["label"] != "Background":
                    polygon = region_dict['shape_attributes']
                    x_points = np.asarray(polygon['all_points_x'])
                    y_points = np.asarray(polygon['all_points_y'])
                    points = np.asarray(list(zip(x_points, y_points)))
                    cv2.fillPoly(mask, np.asarray([points], dtype=np.int32), 1)

            # cutout background parts INSIDE forground 
            for _, region_dict in label_dict["regions"].items():
                if region_dict['region_attributes']["label"] == "Background":
                    polygon = region_dict['shape_attributes']
                    x_points = np.asarray(polygon['all_points_x'])
                    y_points = np.asarray(polygon['all_points_y'])
                    points = np.asarray(list(zip(x_points, y_points)))
                    cv2.fillPoly(mask, np.asarray([points], dtype=np.int32), 0)

            self.bg_masks.append(mask)

    def __len__(self):
        return len(self.filenames)# + len(self.bgs)

    def __getitem__(self, idx):


        if idx < 2 or not self.augment_with_other_bgs:
            img_name = self.filenames[idx]
            mask = self.bg_masks[idx]
            true_fgr = Image.open(img_name).convert("RGB").resize((786, 650))
            true_pha = Image.fromarray((mask * 255).astype(np.uint8)).resize((786, 650))
            true_bgr = true_fgr
        else:
            foreground_idx = random.randrange(1)
            img_name = self.filenames[foreground_idx]
            mask = self.bg_masks[foreground_idx]
            true_fgr = Image.open(img_name).convert("RGB").resize((786, 650))
            true_pha = Image.fromarray((mask * 255).astype(np.uint8)).resize((786, 650))
            true_bgr = Image.open(self.bgs[idx-2]).convert("RGB").resize((786, 650))
            


        sample = dict(true_fgr=true_fgr, true_pha=true_pha, true_bgr=true_bgr)#, true_bgr_pha=true_bgr_pha)

        if self.transform is not None:
            sample = {k: self.transform(v) for k, v in sample.items()}
        
        # test_set = self.vid[random.randrange(1000)]

        # for key, value in sample.items():
        #     sample[key] = torch.cat((value, test_set[key]), dim=0)
        return sample


    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask
    

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
