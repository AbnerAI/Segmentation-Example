import os
import pandas as pd
import random

path = '/Volumes/Data/raw_data/medical_exam/imgs'
imgs_path = os.listdir(path)
idx = [item for item in imgs_path]
random.shuffle(idx)
length = len(idx)
train_set = idx[:int(length*0.8)]
val_set = idx[int(length*0.8):]

train_frame = pd.DataFrame({
    'id': train_set
})
val_frame = pd.DataFrame({
    'id': val_set
})
train_frame.to_csv(f'./exam_train_set_{len(train_set)}.csv', index=False, sep=',')
val_frame.to_csv(f'./exam_val_set_{len(val_set)}.csv', index=False, sep=',')
