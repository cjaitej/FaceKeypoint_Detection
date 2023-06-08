import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import matplotlib.pyplot as plt

def draw_loss_graph(file_name, save_name, from_epoch = 0):
    f = open(file_name, 'r')
    f = f.readlines()
    x = []
    y = []
    for i in f[from_epoch:]:
        temp = i.split(" ")
        x.append(int(temp[1]))
        y.append(float(temp[-1].replace("\n", "")))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x, y)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.savefig(save_name)
    plt.close()


def create_datalist(folder, image_folder, output_folder='./'):
    f = open(folder)
    dict_data = json.load(f)

    data = []

    for i in dict_data:
        temp = []
        temp.append(os.path.join(image_folder, dict_data[i]['file_name']))
        temp.append(dict_data[i]['face_landmarks'])
        data.append(temp)


    with open(os.path.join(output_folder, 'train'+ '_data.json'), 'w') as f:
        json.dump(data, f)

def add_result(result):
    with open('results_v4.txt', 'a') as f:
        f.write(result + "\n")
    f.close()


def transform(image, keypoints, size, split = 'train'):
    if split == 'test':
        transform = A.Compose([
        A.Resize(height=size, width=size),
        # A.Normalize(
        # mean=[0,0,0],
        # std=[1,1,1],
        # ),
        ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy')
        )
    else:
        transform = A.Compose([
            A.Resize(height=size, width=size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),
            # A.Normalize(
                # mean=[0,0,0],
                # std=[1,1,1],
            # ),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy')
        )
    try:
        augmentation = transform(image=image, keypoints=keypoints)
        return augmentation["image"]/255., torch.tensor(augmentation["keypoints"])/size
    except:
        return None, None

def save_checkpoint(epoch, model):
    state = {'epoch': epoch,
             'model': model}
    filename = 'V4_checkpoint_detection.pth.tar'
    torch.save(state, filename)
    # print("Model Saved :)")
