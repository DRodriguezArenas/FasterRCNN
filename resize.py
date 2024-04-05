import os
import cv2 # type: ignore
from torchvision.transforms.functional import pad # type: ignore

def resize_images_masks(img_dir, mask_dir):
    images = os.listdir(img_dir) 
    masks = os.listdir(mask_dir)
    
    for image in images:
        img = cv2.imread(img_dir + image)
        img = cv2.resize(img, (1024, 1024))
        cv2.imwrite('data/Images/' + image, img)
        
    for mask in masks:
        msk = cv2.imread(mask_dir + mask)
        msk = cv2.resize(msk, (1024, 1024))
        cv2.imwrite('data/masks/' + mask, msk)
        

def resize_img(img): 
    