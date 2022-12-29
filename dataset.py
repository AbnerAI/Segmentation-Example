import itertools
import pandas as pd
import os
import cv2
import numpy as np
import image_augment as aug
import torch
from torch.utils.data import Dataset


class GuideDataset(Dataset):

    def __init__(self, imgs_dir, masks_dir, guide_file, size=224, train=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.size = size
        self.train = train
        csv = pd.read_csv(guide_file)
        self.idx = [item[0] for item in csv.values]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        name = self.idx[i]
        img = cv2.imread(os.path.join(self.imgs_dir, name), 0)
        mask = cv2.imread(os.path.join(self.masks_dir, name), 0)
        if img is None or mask is None:
            print(name)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(img, mask, size=self.size)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

    def preprocess(self, np_img, np_mask, size = 224, cmap='gray'):
        newW, newH = size, size
        np_img = cv2.resize(np_img, (newW, newH))
        np_mask = cv2.resize(np_mask, (newW, newH))

        if cmap == 'rgb' :
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)

        if len(np_img.shape) == 2:
            np_img = np.expand_dims(np_img, axis=2)
        if len(np_mask.shape) == 2:
            np_mask = np.expand_dims(np_mask, axis=2)


        # HWC to NHWC for augmentation
        img_trans = np.expand_dims(np_img, axis=0)
        mask_trans = np.expand_dims(np_mask, axis=0)/255.0

        # augmentation
        if self.train :
            img_aug, mask_aug = aug.seq(images=img_trans.astype(np.uint8), heatmaps=mask_trans.astype(np.float32))
            img_aug = np.array(img_aug)
            mask_aug = np.array(mask_aug)
        else :
            img_aug, mask_aug = img_trans, mask_trans
        # NHWC to CHW
        img_trans = img_aug.reshape(img_aug.shape[1:]).transpose((2, 0, 1))
        mask_trans = mask_aug.reshape(mask_aug.shape[1:]).transpose((2, 0, 1))

        return img_trans/255.0, mask_trans