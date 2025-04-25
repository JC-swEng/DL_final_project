"References: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html"
""

import os
from PIL import Image
from torch.utils.data import Dataset

class AffectnetYoloDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to 'train', 'valid', or 'test' folder
        transform: Any torchvision transforms (optional) https://pytorch.org/vision/main/transforms.html
        """
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform

        self.data = []
        for label_file in os.listdir(self.label_dir):
            label_path = os.path.join(self.label_dir, label_file)
            with open(label_path, 'r') as f:
                label = int(f.readline().split()[0])  # first number in txt

            base_name = os.path.splitext(label_file)[0]
            img_path_jpg = os.path.join(self.image_dir, base_name + ".jpg")
            img_path_png = os.path.join(self.image_dir, base_name + ".png")

            #jpg/png
            if os.path.exists(img_path_jpg):
                img_path = img_path_jpg
            elif os.path.exists(img_path_png):
                img_path = img_path_png
            else:
                continue  # Skip if no image found

            self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label