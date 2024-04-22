import os
import cv2 # type: ignore
import numpy as np # type: ignore
import torchvision.transforms.functional as TF # type: ignore
from PIL import Image # type: ignore

def resize_images_masks(img_dir, mask_dir, target_height, target_width):
    total_images = os.listdir(img_dir) 
    total_masks = os.listdir(mask_dir)
        
    # for image in total_images:
    #     img = cv2.imread(img_dir + image)
    #     height, width, _ = img.shape
    #     proportion = width / height
    #     final_img_height = int(target_width / proportion)
    #     img_resized = cv2.resize(img, (target_width, final_img_height))
    #     pad_height = target_height - final_img_height
    #     pad_top = pad_height // 2
    #     pad_bottom = pad_height - pad_top      
    #     img_pil = Image.fromarray(img_resized)
    #     img_padded = TF.pad(img_pil, (0, pad_top, 0, pad_bottom), fill=0)
    #     img_padded = np.array(img_padded)

    #     cv2.imwrite('data/Images_resized/' + image, img_padded)
    
    for mask_path in total_masks:
        mask = np.load(mask_dir + mask_path, allow_pickle=True)
        masks = mask['obj']
        resized_masks = []
        for mask_info in masks:
            mask = mask_info[1]
            mask = np.array(mask)
            height, width = mask.shape
            proportion = width / height
            final_msk_height = int(target_width / proportion)
            msk_resized = cv2.resize(mask, (target_width, final_msk_height))
            pad_height = target_height - final_msk_height
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            msk_resized = Image.fromarray(msk_resized)
            msk_padded = TF.pad(msk_resized, (0, pad_top, 0, pad_bottom), fill=0)
            msk_padded = np.array(msk_padded)
            resized_masks.append(('strawberry', msk_padded))
        np.savez_compressed('data/masks_resized/' + mask_path, obj=np.array(resized_masks, dtype=object))

        