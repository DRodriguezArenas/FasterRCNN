import torch # type: ignore
import os
from torch.utils.data import Dataset # type: ignore
import numpy as np # type: ignore
from PIL import Image, ImageDraw # type: ignore
import torchvision.transforms as T  # type: ignore
from utils.masks_bb import masks_bb, show_masks_bb
 
class PearDataset(Dataset):
    def __init__(self, image_dir, masks_dir, transforms=None):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
    def __len__(self):
        return len(self.image_dir)
    def __getitem__(self, idx):
        img_dir = os.listdir(self.image_dir)
        print("Image dir:" + img_dir)
        img_path = img_dir[idx]
        img = T.toTensor(Image.open(self.image_dir + img_path).convert("RGB"))
        mask_dir = os.listdir(self.masks_dir)
        print("Mask dir:" + mask_dir)
        mask_path = mask_dir[idx]
        boxes, masks = masks_bb(mask_path)
        num_objs = len(boxes)
        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target
    def show_image(self, idx):
        img_dir = os.listdir(self.image_dir)
        img_path = img_dir[idx]
        mask_dir = os.listdir(self.masks_dir)
        mask_path = mask_dir[idx + 2]
        show_masks_bb(self.image_dir + img_path, self.masks_dir + mask_path) 