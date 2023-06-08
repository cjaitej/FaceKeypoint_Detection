from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
import torch
from utils import transform

class FaceKeypoints(Dataset):
    def __init__(self, file_name, size, transform=None, split='train'):
        super(FaceKeypoints, self).__init__()
        f = open(file_name)
        self.data = json.load(f)
        self.size = size
        self.split = split
        if split == 'test':
            self.data = self.data[4000:]
        else:
            self.data = self.data[:4000]

        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index][0]
        keypoints = self.data[index][1]

        image = np.array(Image.open(image_path).convert('L')).reshape(512, -1, 1)
        keypoints = np.array(keypoints)

        if self.transform:
            image, keypoints = self.transform(image, keypoints, self.size, split=self.split)
        if image is None or keypoints.shape[0]*keypoints.shape[1] != 136:
            return None, None
        keypoints = keypoints.permute(1, 0)
        keypoints = keypoints.reshape(-1)    # x coordinates followed by y coordinates.
        return image.to(torch.float32), keypoints.to(torch.float32)

    def collate_fn(self, batch):
        images = []
        keypoints = []

        for b in batch:
            if b[0] is None:
                continue
            images.append(b[0])
            keypoints.append(b[1])
        images = torch.stack(images, dim=0)
        keypoints = torch.stack(keypoints, dim=0)
        return images, keypoints

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = FaceKeypoints('train_data.json', size=128, transform=transform)
    for i, j in dataset:
        print(j.shape)
        break
