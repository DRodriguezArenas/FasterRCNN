import torch # type: ignore
import os
from torch.utils.data import Dataset # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import torchvision.transforms as T  # type: ignore
from utils.masks_bb import masks_bb, show_masks_bb
import matplotlib.pyplot as plt # type: ignore
import cv2 # type: ignore
import random
import matplotlib.colors as mcolors

class StrawberryDataset(Dataset):
    def __init__(self, image_dir, masks_dir, boxes, masks, transforms=None):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.boxes = boxes
        self.masks = masks
        self.transforms = transforms
    def __len__(self):
        img_dir = os.listdir(self.image_dir)
        return len(img_dir)
    def __getitem__(self, idx):
        img_dir = os.listdir(self.image_dir)
        img_path = img_dir[idx]
        img = T.ToTensor()(Image.open(self.image_dir + img_path).convert("RGB"))
        num_objs = len(self.boxes)
        boxes = np.array([[int(num) for num in tensor.squeeze().tolist()] for tensor in self.boxes[idx]])
        masks = [tensor.bool().tolist() for tensor in self.masks[idx]]
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    def generate_random_colors(self, num_colors):
        colors = []
        for _ in range(num_colors):
            rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            pil_string = '#%02x%02x%02x' % rgb
            colors.append(pil_string)
        return colors
    def show_image(self, image, masks, boxes):
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()

        num_colors = len(masks)
        colors = self.generate_random_colors(num_colors)

        for mask, color in zip(masks, colors):
            mask_np = mask.squeeze(0).detach().cpu().numpy()
            mask_binary = np.where(mask_np > 0.5, 1, 0)
            rgb_color = mcolors.to_rgb(color)
            image_np[mask_binary == 1] = np.array(rgb_color) * 0.7 + np.array(image_np[mask_binary == 1]) * 0.3

        for box, color in zip(boxes, colors):
            x1, y1, x2, y2 = map(int, box.tolist())
            rgb_color = mcolors.to_rgb(color)
            image_np = cv2.rectangle(image_np.copy(), (x1, y1), (x2, y2), rgb_color, 4)

        plt.imshow(image_np)
        plt.axis('off')
        plt.show()


