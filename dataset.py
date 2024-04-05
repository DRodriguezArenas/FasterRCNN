import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
from PIL import Image, ImageDraw # type: ignore
import torchvision.transforms as T  # type: ignore
 
class WheatDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = f"{self.image_dir}/{img_name}"
        img = T.PILToTensor()(Image.open(img_path).convert("RGB"))/255
        boxes = self.df.iloc[idx, 1]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, img_name
    def show_image(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = f"{self.image_dir}/{img_name}"
        img = Image.open(img_path).convert("RGB")
        boxes = self.df.iloc[idx, 1]
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red")
        return img
 