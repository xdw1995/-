""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data_augument import *
DATAPATH = '/home/work/CUB_200_2011'
image_path = {}
image_label = {}


class BirdDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', transform = None):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.image_id = []
        self.num_classes = 200

        # get image path from images.txt
        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

        # transform
        self.transform = transform

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(DATAPATH, 'images', image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, image_label[image_id] - 1  # count begin from zero

    def __len__(self):
        return len(self.image_id)


if __name__ == '__main__':
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(448, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    ImageNetPolicy(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
    [0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = BirdDataset('train',transform= train_transform)

    val_loader = BirdDataset('val',transform= val_transform)
    print(len(train_loader))
    for i in range(0, 10):
        image, label = train_loader[i]
        print(image.shape, label)
