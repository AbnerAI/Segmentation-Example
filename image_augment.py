'''
数据增强
'''

import imgaug.augmenters as iaa
import cv2
import numpy as np
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([

    # # 随机噪点或模糊
    # sometimes(iaa.OneOf([
    #     iaa.GaussianBlur((0, 3.0)),
    #     iaa.Dropout(0.1),
    # ])),
    #
    # # 随机变换
    # sometimes(iaa.Crop(px=(0, 10))),
    # sometimes(iaa.Affine(rotate=(-45, 45))),

    # 弹性形变
    sometimes(iaa.ElasticTransformation(alpha=(0, 10), sigma=1.0)),
    iaa.Fliplr(0.5),
    sometimes(iaa.Rot90(1))

])


if __name__ == '__main__':
    img = cv2.imread('./data/imgs/18.PNG')
    mask = cv2.imread('./data/masks/18.PNG',0)
    img = cv2.resize(img, (512,512))
    mask = cv2.resize(mask, (512,512))
    cv2.imshow('oimg', img)
    cv2.imshow('omask', mask)
    img = np.expand_dims(img, axis=0)
    mask = np.expand_dims(mask, axis=2)
    mask = np.expand_dims(mask, axis=0)
    img, mask = seq(images=img.astype(np.uint8), heatmaps=(mask.astype(np.float32)/255.0))
    img = img.reshape(img.shape[1:])
    mask = mask.reshape(mask.shape[1:3])
    cv2.imshow('aimg', img)
    cv2.imshow('amask', mask)
    cv2.waitKey()
