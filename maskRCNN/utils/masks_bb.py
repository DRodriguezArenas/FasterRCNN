import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import os
import torch # type: ignore
import cv2 # type: ignore
import torchvision.transforms.functional as F # type: ignore
from torchvision.io import read_image # type: ignore
from torchvision.ops import masks_to_boxes # type: ignore
from torchvision.utils import draw_segmentation_masks # type: ignore
from torchvision.utils import draw_bounding_boxes # type: ignore
import random

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def generate_random_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pil_string = '#%02x%02x%02x' % rgb
        colors.append(pil_string)
    return colors

def masks_bb(masks_path):
    masks_dir = os.listdir(masks_path)

    total_boxes = []
    total_masks = []

    for mask_path in masks_dir:
        masks = np.load(masks_path + mask_path, allow_pickle=True)
        masks = masks['obj']

        group_boxes = []
        group_masks = []

        for mask in masks:
            mask = F.to_tensor(mask[1])
            mask = mask.to(dtype=torch.uint8)
            
            obj_ids = torch.unique(mask)
            obj_ids = obj_ids[1:]
            masks = mask == obj_ids[:, None, None]

            drawn_mask = masks.squeeze(0).bool()
            group_masks.append(drawn_mask)
            
            boxes = masks_to_boxes(masks)
            group_boxes.append(boxes)

        total_boxes.append(group_boxes)
        total_masks.append(group_masks)
    
    return total_masks, total_boxes

def show_masks_bb(img, masks, boxes):
    colors = generate_random_colors(len(masks))

    masks = masks.squeeze(1)  
    masks = masks.bool()
    drawn_masks = torch.stack((masks,), dim=0)
    drawn_masks = draw_segmentation_masks(img, masks, alpha=0.6, colors=colors)
    show(drawn_masks)
    
    if img.dtype != torch.uint8:
        img_min = img.min()
        img_max = img.max()
        img = ((img - img_min) / (img_max - img_min) * 255).to(torch.uint8)
        
    total_boxes = torch.cat((boxes,), dim=0)
    drawn_boxes = draw_bounding_boxes(img, total_boxes, colors=colors, width=5)
    show(drawn_boxes)