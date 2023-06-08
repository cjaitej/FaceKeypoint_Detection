from model import Detection
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FaceKeypoints
from utils import *
from torch import optim

def train(checkpoint):
    if checkpoint == None:
        model = Detection(in_c=1)
        start_epoch = 0
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f'\n -- Loaded checkpoint from epoch {start_epoch}--\n')
        model = checkpoint['model']
    model = model.to(device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    average_loss = 0
    print(" --- Starting Training ---")
    for epoch in range(start_epoch, epochs):
        for i, (image, keypoints) in enumerate(train_loader):
            image = image.to(device)
            keypoints = keypoints.to(device)
            pred_keypoints = model(image)
            loss = criterion(pred_keypoints, keypoints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss = average_loss + loss
            del image, keypoints, pred_keypoints
            if i%10 == 0:
                print("=", end="")
        save_checkpoint(epoch, model)
        add_result(f"Epoch: {epoch} | Average Loss: {average_loss/(i + 1)}")
        print(f"   Epoch: {epoch} | Average Loss: {average_loss/(i + 1)}")
        average_loss = 0




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "V4_checkpoint_detection.pth.tar"
    batch_size = 100
    iterations = 120000
    workers = 4
    epochs = 1000
    print_freq = 200
    lr = 0.00001
    file_name = 'train_data.json'

    train_dataset = FaceKeypoints(file_name=file_name, size=128, transform=transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=workers,
                          collate_fn=train_dataset.collate_fn,
                          pin_memory=True)
    train(checkpoint)