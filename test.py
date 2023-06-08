import torch
from dataset import FaceKeypoints
from utils import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "V4_checkpoint_detection.pth.tar"
    file_name = 'train_data.json'

    train_dataset = FaceKeypoints(file_name=file_name, size=128, transform=transform, split='test')


    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device=device)

    with torch.inference_mode():
        for image, keypoints in train_dataset:
            image = image.to(device)
            image = image.unsqueeze(0)
            out = model(image)
            out = out.reshape(2, -1)*128
            keypoints = keypoints.reshape(2, -1)*128
            image = image.squeeze(0).permute(1, 2, 0).cpu()
            out = out.cpu()
            keypoints = keypoints.cpu()
            print(keypoints.unique())
            plt.imshow(image)
            plt.scatter(out[0], out[1], c='red')
            plt.scatter(keypoints[0], keypoints[1], c='yellow', s=10)
            plt.show()

# average_loss = 0