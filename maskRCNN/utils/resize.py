import os
import cv2 # type: ignore
import numpy as np # type: ignore
import torchvision.transforms.functional as TF # type: ignore
from PIL import Image # type: ignore

def resize_images_masks(img_dir, mask_dir, target_height, target_width):
    total_images = os.listdir(img_dir) 
    total_masks = os.listdir(mask_dir)
        
    for image in total_images:
        if image != '20220629_R1T17_PRF_130_PIC2.jpg':
            continue
        img = cv2.imread(img_dir + image)
        height, width, _ = img.shape
    
        proportion = height / width
                
        final_img_width = int(target_height / proportion)

        img_resized = cv2.resize(img, (final_img_width, target_height))
        
        pad_width = target_width - final_img_width
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
                
        img_pil = Image.fromarray(img_resized)
        img_padded = TF.pad(img_pil, (pad_left, 0, pad_right, 0), fill=0)
        img_padded = np.array(img_padded)

        cv2.imwrite('20220629_R1T17_PRF_130_PIC2.jpg', img_padded)
        break
    
    for mask in total_masks:
        if mask != '20220629_R1T17_PRF_130_PIC2.npz':
            continue
        mask = np.load(mask_dir + mask, allow_pickle=True)
        masks = mask['obj']
        resized_masks = []
        for mask_info in masks:
            mask = mask_info[1]
            mask = np.array(mask)
            height, width = mask.shape
            proportion = height / width
            final_msk_width = int(target_height / proportion)
            msk_resized = cv2.resize(mask, (final_msk_width, target_height))
            pad_width = target_width - final_msk_width
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            msk_resized = Image.fromarray(msk_resized)
            msk_padded = TF.pad(msk_resized, (pad_left, 0, pad_right, 0), fill=0)
            msk_padded = np.array(msk_padded)
            resized_masks.append(('pear', msk_padded))
            
        np.savez('20220629_R1T17_PRF_130_PIC2.npz', obj=np.array(resized_masks, dtype=object))
        break
    
