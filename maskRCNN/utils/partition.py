import os 
import random
import shutil

def partition_data(src_images_dir, src_masks_dir, train_images_dir, train_masks_dir, test_images_dir, test_masks_dir, split_ratio=0.9):
    image_files = os.listdir(src_images_dir)
    mask_files = os.listdir(src_masks_dir)
    
    num_train = int(len(image_files) * split_ratio)
    
    random.shuffle(image_files)
    
    train_images = image_files[:num_train]
    train_masks = [filename.replace('.jpg', '.npz') for filename in train_images]
    
    test_images = image_files[num_train:]
    test_masks = [filename.replace('.jpg', '.npz') for filename in test_images]
    
    for img, msk in zip(train_images, train_masks):
        shutil.move(os.path.join(src_images_dir, img), os.path.join(train_images_dir, img))
        shutil.move(os.path.join(src_masks_dir, msk), os.path.join(train_masks_dir, msk))
        
    for img, msk in zip(test_images, test_masks):
        shutil.move(os.path.join(src_images_dir, img), os.path.join(test_images_dir, img))
        shutil.move(os.path.join(src_masks_dir, msk), os.path.join(test_masks_dir, msk))