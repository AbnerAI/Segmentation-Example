import argparse
import os
import torch
import torch.nn as nn
import math
import random
import shutil
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from common import save_checkpoint
from dataset import GuideDataset

def evaluate(model, loader, device):

    '''
    TP(true positive), TN(true negative), FP(false postive), FN(false negative)
    AC = (TP + TN)/(TP + FP + TN + FN)
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    JA = TP / (TP + FN + FP)
    DI = 2 * TP / (2 * TP + FN + FP)
    :return:
    ac,se,sp,ja,di
    '''

    tp, tn, fp, fn = 0, 0, 0, 0
    model.eval()
    for sample in loader:
        imgs, labels = sample['image'], sample['mask']
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            res = model(imgs)
        pred = (torch.sigmoid(res) > 0.5).float()
        cmp = pred + labels
        tp_tmp = (cmp > 1).sum()
        tn_tmp = (cmp < 1).sum()
        fp_tmp = (((pred == 1).float() + (labels == 0).float()) == 2).sum()
        fn_tmp = (((pred == 0).float() + (labels == 1).float()) == 2).sum()
        tp += tp_tmp
        tn += tn_tmp
        fp += fp_tmp
        fn += fn_tmp

    ac = (tp + tn) / (tp + fp + tn + fn)
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    ja = tp / (tp + fn + fp)
    di = 2 * tp / (2 * tp + fn + fp)
    model.train()

    return ac, se, sp, ja, di

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32, dest='batch_size')
    parser.add_argument('-l', '--lr', type=float, default=3e-4, dest='lr')
    parser.add_argument('-i', '--iteration', type=int, default=8000, dest='iteration')
    parser.add_argument('-n', '--exp_name', type=str, default='exp', dest='exp_name')
    parser.add_argument('--labeled_num', type=int, default=600, dest='labeled_num')
    return parser.parse_args()


def train(train_loader, val_loader, model, optimizer, epochs, device, writer):

    # ---- prepare ----
    model.to(device)
    model.train()
    total_step = 0
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=8, factor=0.1, threshold=0.0001)
    best_performance = 0

    # ---- training ----
    for epoch in range(1, epochs+1):
        with tqdm(total=len(train_loader), desc=f'epoch[{epoch}/{epochs+1}]:') as pbar:
            for i, batch in enumerate(train_loader):

                # ---- data prepare ----
                images, labels = batch['image'], batch['mask']
                images = images.to(device)
                labels = labels.to(device)

                # ---- forward ----
                preds = model(images)
                # ---- loss ----
                loss = criterion(preds, labels)

                # ---- backward ----
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_step += 1

                # ---- log ----
                writer.add_scalar('info/loss', loss, total_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

                # ---- validation ----
                if total_step % 200 == 0:
                    _, _, _, ja, di = evaluate(model, val_loader, device)
                    if ja > best_performance:
                        best_performance = ja
                    scheduler.step(ja)
                    writer.add_scalar('eval/JA', ja, total_step)
                    writer.add_scalar('eval/DI', di, total_step)
                    writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], total_step)
                    print(f"""
                        performance {ja}
                    """)

    print('training finished')

if __name__ == '__main__':

    # ---- common setting ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    imgs_dir = '/home/work/data/yzy/data/medical_exam/imgs'
    labels_dir = '/home/work/data/yzy/data/medical_exam/masks'
    train_guide_file_path = 'exam_train_set_1590.csv'
    val_guide_file_path = 'exam_val_set_398.csv'
    log_path = '/home/work/data/yzy/log/'

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    torch.cuda.manual_seed(999)

    # ---- build model ----
    model = smp.Unet("efficientnet-b5", in_channels=1, classes=1)

    # ---- dataset ----
    train_set = GuideDataset(imgs_dir, labels_dir, train_guide_file_path, size=224, train=True)
    val_set = GuideDataset(imgs_dir, labels_dir, val_guide_file_path, size=224)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    # ---- experiment option ----
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    epochs = math.ceil(args.iteration / len(train_loader))
    if os.path.exists(os.path.join(log_path, args.exp_name, 'log')):
        shutil.rmtree(os.path.join(log_path, args.exp_name, 'log'))
    writer = SummaryWriter(os.path.join(log_path, args.exp_name, 'log'))

    # ---- train ____
    print('training start!')

    train(train_loader, val_loader, model, optimizer, epochs, device, writer)

    ac, se, sp, ja, di = evaluate(model, val_loader, device)

    print(f'''
        test performence:
        Jaccard index(JA, iou):         {ja}
        Dice coefficient(DI):           {di}
        Pixelwise accuracy(AC):         {ac}
        Sensitivity(SE):                {se}
        Specificity(SP):                {sp}
        ''')

    save_checkpoint(model, os.path.join(log_path, args.exp_name, 'checkpoints'), 0, ja)






