import os
import cv2
import torch
import pandas as pd
import segmentation_models_pytorch as smp

model_path = '/home/work/data/yzy/log/exam/checkpoints/CP_0_0.6877.pth'
model = smp.Unet('efficientnet-b5', in_channels=1, classes=1).cuda()
model.load_state_dict(torch.load(model_path))

train = pd.read_csv('exam_train_set_1590.csv')
val = pd.read_csv('exam_val_set_398.csv')
imgs_dir = '/home/work/data/yzy/data/medical_exam/imgs'

def predict(name):
    img = cv2.imread(os.path.join(imgs_dir, name), 0)
    w,h = img.shape
    img = cv2.resize(img, (224, 224))
    _input = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    _input = _input / 255.0
    _input = _input.cuda()
    with torch.no_grad():
        pred = model(_input)
        pred = (torch.sigmoid(pred) > 0.5).float() * 255.0
    pred_np = pred.squeeze().cpu().numpy()
    pred_np = cv2.resize(pred_np, (h, w))
    return pred_np
if not os.path.exists('exam_train_prediction'):
    os.makedirs('exam_train_prediction')
if not os.path.exists('exam_val_prediction'):
    os.makedirs('exam_val_prediction')
count = 0
for item in train.values:
    pred = predict(item[0])
    cv2.imwrite('exam_train_prediction/' + item[0], pred)
    count += 1
    print(count)

for item in val.values:
    pred = predict(item[0])
    cv2.imwrite('exam_val_prediction/' + item[0], pred)
    count += 1
    print(count)