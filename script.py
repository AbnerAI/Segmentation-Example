import os
import shutil
import pandas as pd
import cv2
import numpy as np


def visualize(img_path, mask_path, pred_path, out_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    pred = cv2.imread(pred_path, 0)

    pred_color = (189, 255, 204)
    mask_color = (69, 69, 232)

    _, mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, pred_contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, mask_contours, -1, mask_color, 3)
    board = np.zeros_like(img)
    cv2.drawContours(board, pred_contours, -1, (255, 255, 255), 3)
    board = board/255.0 * 0.5
    img = (1 - board) * img + board * np.array(pred_color)
    cv2.imwrite(out_path, img)


imgs_path = '/Volumes/Data/raw_data/xiangya_video_clips/imgs'
mask_path = '/Volumes/Data/raw_data/xiangya_video_clips/masks'
root = '/Volumes/Data/结果/湘雅/'

train_csv = pd.read_csv('xiangya_train_set_2878.csv')
test_csv = pd.read_csv('xiangya_val_set_720.csv')

# 金标准
gt_path = os.path.join(root, '金标准')
if not os.path.exists(os.path.join(gt_path, '训练集')):
    os.makedirs(os.path.join(gt_path, '训练集'))
if not os.path.exists(os.path.join(gt_path, '测试集')):
    os.makedirs(os.path.join(gt_path, '测试集'))

for item in train_csv.values:
    shutil.copy(
        os.path.join(mask_path, item[0]),
        os.path.join(gt_path, '训练集', item[0])
    )

for item in test_csv.values:
    shutil.copy(
        os.path.join(mask_path, item[0]),
        os.path.join(gt_path, '测试集', item[0])
    )

# 原图
gt_path = os.path.join(root, '超声图')
if not os.path.exists(os.path.join(gt_path, '训练集')):
    os.makedirs(os.path.join(gt_path, '训练集'))
if not os.path.exists(os.path.join(gt_path, '测试集')):
    os.makedirs(os.path.join(gt_path, '测试集'))

for item in train_csv.values:
    shutil.copy(
        os.path.join(imgs_path, item[0]),
        os.path.join(gt_path, '训练集', item[0])
    )

for item in test_csv.values:
    shutil.copy(
        os.path.join(imgs_path, item[0]),
        os.path.join(gt_path, '测试集', item[0])
    )

# 可视化
gt_path = os.path.join(root, '可视化')
if not os.path.exists(os.path.join(gt_path, '训练集')):
    os.makedirs(os.path.join(gt_path, '训练集'))
if not os.path.exists(os.path.join(gt_path, '测试集')):
    os.makedirs(os.path.join(gt_path, '测试集'))

for i, item in enumerate(train_csv.values):
    visualize(
        os.path.join(root, '超声图', '训练集', item[0]),
        os.path.join(root, '金标准', '训练集', item[0]),
        os.path.join(root, '模型预测', '训练集', item[0]),
        os.path.join(root, '可视化','训练集', item[0])
    )
    print('train', i)

for i, item in enumerate(test_csv.values):
    visualize(
        os.path.join(root, '超声图', '测试集', item[0]),
        os.path.join(root, '金标准', '测试集', item[0]),
        os.path.join(root, '模型预测', '测试集', item[0]),
        os.path.join(root, '可视化', '测试集', item[0])
    )
    print('test', i)

