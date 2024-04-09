import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
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
        # Generar un color aleatorio en formato RGB
        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # Convertir el color RGB en formato PIL string
        pil_string = '#%02x%02x%02x' % rgb
        colors.append(pil_string)
    return colors

def masks_bb(masks_path):
    masks = np.load(masks_path, allow_pickle=True)
    masks = masks['obj']

    total_boxes = []
    drawn_masks = []

    for mask in masks:
        mask = F.to_tensor(mask[1])
        mask = mask.to(dtype=torch.uint8)
        
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        drawn_mask = masks.squeeze(0).bool()
        drawn_masks.append(drawn_mask)
        
        boxes = masks_to_boxes(masks)
        
        total_boxes.append(boxes)
    
    return total_boxes, drawn_masks

def show_masks_bb(img_path, masks_path):
    print(img_path, masks_path)
    img = cv2.imread(img_path)
    img = F.to_tensor(img)
    img = img.mul(255).byte()  
    masks = np.load(masks_path, allow_pickle=True)
    masks = masks['obj']

    total_boxes = []
    drawn_masks = []
    
    total_boxes, drawn_masks = masks_bb(masks_path)
    colors = generate_random_colors(len(total_boxes))

    drawn_masks = torch.stack(drawn_masks, dim=0)
    drawn_masks = draw_segmentation_masks(img, drawn_masks, alpha=0.6, colors=colors)
    show(drawn_masks)

    total_boxes = torch.cat(total_boxes)
    drawn_boxes = draw_bounding_boxes(img, total_boxes, colors=colors, width=5)
    show(drawn_boxes)
