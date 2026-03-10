#!/usr/bin/env python3
"""
train_Twitter.py

Trains the AUG model (TIGBShareClassifier) on Twitter15 / Twitter17.

Usage:
    python train_Twitter.py --config ./data/twitter.json

Mirrors train_CREMAD.py 1-to-1:
  - audio  → text   (is_a=True  branch)
  - video  → image  (is_a=False branch)
  - SGD    → AdamW  (BERT requires adaptive optimiser; two param groups)
  - All AUG logic (ratio_a, add_layer, merge_alpha) is identical.
"""

import os
import json
import random
import warnings
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score

warnings.filterwarnings("ignore")

from data.template import config
from dataset.Twitter import TwitterDataset
from model.TextImage import TIGBShareClassifier
from utils.utils import Averager, deep_update_dict, weight_init


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


# ------------------------------------------------------------------ #
#  Training epoch                                                      #
# ------------------------------------------------------------------ #

def train_text_image(epoch, train_loader, model, optimizer, merge_alpha=0.5):
    model.train()

    tl = Averager()
    ta = Averager()   # text  loss tracker
    tv = Averager()   # image loss tracker

    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    score_t = 0.0   # running softmax score — text  branch (mirrors score_a)
    score_i = 0.0   # running softmax score — image branch (mirrors score_v)

    for step, (input_ids, attention_mask, token_type_ids, image, y) in enumerate(train_loader):

        input_ids      = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()
        image          = image.float().cuda()
        y              = y.cuda()

        # ---- TEXT branch (mirrors audio branch) -------------------------
        optimizer.zero_grad()

        o_t = model.text_encoder(input_ids, attention_mask, token_type_ids)
        out_t, o_fea, add_fea = model.classfier(o_t, is_a=True)

        if add_fea is None:
            loss_t = criterion(out_t, y).mean()
        else:
            kl     = y * o_fea.detach().softmax(1)
            loss_t = (criterion(out_t, y).mean()
                      + criterion(o_fea, y).mean()
                      + criterion(add_fea, y).mean()
                      - 0.5 * criterion(add_fea, kl).mean())

        loss_t.backward()
        optimizer.step()
        optimizer.zero_grad()

        # ---- IMAGE branch (mirrors video branch) ------------------------
        o_i = model.image_encoder(image)
        out_i, o_fea, add_fea = model.classfier(o_i, is_a=False)

        if add_fea is None:
            loss_i = criterion(out_i, y).mean()
        else:
            kl     = y * o_fea.detach().softmax(1)
            loss_i = (criterion(out_i, y).mean()
                      + criterion(o_fea, y).mean()
                      + criterion(add_fea, y).mean()
                      - 0.5 * criterion(add_fea, kl).mean())

        loss_i.backward()
        optimizer.step()
        optimizer.zero_grad()

        # ---- Bookkeeping ------------------------------------------------
        loss = loss_t * merge_alpha + loss_i * (1 - merge_alpha)
        tl.add(loss.item())
        ta.add(loss_t.item())
        tv.add(loss_i.item())

        tmp_i = sum([F.softmax(out_i, dim=1)[i][torch.argmax(y[i])]
                     for i in range(out_i.size(0))])
        tmp_t = sum([F.softmax(out_t, dim=1)[i][torch.argmax(y[i])]
                     for i in range(out_t.size(0))])
        score_i += tmp_i
        score_t += tmp_t

        # clear any leftover gradients
        for _, p in model.named_parameters():
            if p.grad is not None:
                del p.grad

        if step % cfg.get('print_inteval', 20) == 0:
            print(
                f"Epoch:{epoch}  Loss:{loss.item():.3f}  "
                f"Loss_text:{loss_t.item():.3f}  Loss_img:{loss_i.item():.3f}"
            )

    # ratio_a = score_text / score_image  (mirrors ratio_a in CREMA)
    ratio_a      = score_t / (score_i + 1e-8)
    loss_ave     = tl.item()
    loss_t_ave   = ta.item()
    loss_i_ave   = tv.item()

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(
        f"Epoch {epoch}: Avg Loss:{loss_ave:.3f}  "
        f"Avg Loss_text:{loss_t_ave:.2f}  Avg Loss_img:{loss_i_ave:.2f}"
    )
    return model, ratio_a, loss_ave, loss_t_ave, loss_i_ave


# ------------------------------------------------------------------ #
#  Validation / Test                                                   #
# ------------------------------------------------------------------ #

def val(epoch, val_loader, model, merge_alpha=0.5):
    model.eval()

    pred_list   = []
    pred_list_t = []   # text  preds
    pred_list_i = []   # image preds
    label_list  = []
    soft_pred   = []
    soft_pred_t = []
    soft_pred_i = []
    one_hot_label = []

    with torch.no_grad():
        for step, (input_ids, attention_mask, token_type_ids, image, y) in enumerate(val_loader):

            label_list    += torch.argmax(y, dim=1).tolist()
            one_hot_label += y.tolist()

            input_ids      = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            image          = image.float().cuda()
            y              = y.cuda()

            o_t, o_i = model(input_ids, attention_mask, token_type_ids, image)
            out_t, _, _ = model.classfier(o_t, is_a=True)
            out_i, _, _ = model.classfier(o_i, is_a=False)
            out = merge_alpha * out_t + (1 - merge_alpha) * out_i

            soft_pred_t += F.softmax(out_t, dim=1).tolist()
            soft_pred_i += F.softmax(out_i, dim=1).tolist()
            soft_pred   += F.softmax(out,   dim=1).tolist()

            pred_list   += F.softmax(out,   dim=1).argmax(dim=1).tolist()
            pred_list_t += F.softmax(out_t, dim=1).argmax(dim=1).tolist()
            pred_list_i += F.softmax(out_i, dim=1).argmax(dim=1).tolist()

    f1   = f1_score(label_list, pred_list,   average='macro')
    f1_t = f1_score(label_list, pred_list_t, average='macro')
    f1_i = f1_score(label_list, pred_list_i, average='macro')

    acc   = sum(x == y for x, y in zip(label_list, pred_list))   / len(label_list)
    acc_t = sum(x == y for x, y in zip(label_list, pred_list_t)) / len(label_list)
    acc_i = sum(x == y for x, y in zip(label_list, pred_list_i)) / len(label_list)

    mAP   = compute_mAP(torch.Tensor(soft_pred),   torch.Tensor(one_hot_label))
    mAP_t = compute_mAP(torch.Tensor(soft_pred_t), torch.Tensor(one_hot_label))
    mAP_i = compute_mAP(torch.Tensor(soft_pred_i), torch.Tensor(one_hot_label))

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(
        f"Epoch {epoch}: "
        f"f1:{f1:.4f} acc:{acc:.4f} mAP:{mAP:.4f} | "
        f"f1_text:{f1_t:.4f} acc_text:{acc_t:.4f} mAP_text:{mAP_t:.4f} | "
        f"f1_img:{f1_i:.4f} acc_img:{acc_i:.4f} mAP_img:{mAP_i:.4f}"
    )
    return f1, acc, mAP, f1_t, acc_t, mAP_t, f1_i, acc_i, mAP_i


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      type=str,  default='./data/twitter.json')
    parser.add_argument('--lam',         type=float, default=1.0,  help='AUG ratio threshold')
    parser.add_argument('--merge_alpha', type=float, default=0.5,  help='text branch weight')
    args = parser.parse_args()


    # -----Save Training Logs-----
    model_folder_path = f"Twitter_Models"
    os.makedirs(model_folder_path)

    log_file = open(f"training_logs/Twitter.csv", "w", newline="")
    log_writer = csv.writer(log_file)

    log_writer.writerow([
        "epoch",
        "loss",
        "loss_text",
        "loss_image",
        "f1",
        "acc",
        "mAP",
        "f1_text",
        "acc_text",
        "mAP_text",
        "f1_image",
        "acc_image",
        "mAP_image",
        "ratio_a"
    ])


    cfg = config
    with open(args.config, "r") as f:
        exp_params = json.load(f)
    cfg = deep_update_dict(exp_params, cfg)

    # ---- reproducibility ------------------------------------------------
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']

    # ---- data -----------------------------------------------------------
    train_dataset = TwitterDataset(cfg, mode='train')
    val_dataset   = TwitterDataset(cfg, mode='dev')
    test_dataset  = TwitterDataset(cfg, mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers'],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers'],
        pin_memory=True,
    )

    # ---- model ----------------------------------------------------------
    model = TIGBShareClassifier(config=cfg)
    model = model.cuda()

    # Only apply weight_init to non-BERT layers (BERT has pretrained weights)
    for name, module in model.named_modules():
        if 'bert' not in name:
            weight_init(module)

    # ---- optimiser — two param groups -----------------------------------
    # BERT parameters need a much smaller LR than the randomly-init heads
    bert_params  = list(model.text_encoder.bert.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is bp for bp in bert_params)]

    bert_lr      = cfg['train'].get('bert_lr', 2e-5)
    head_lr      = cfg['train'].get('head_lr', 1e-3)
    weight_decay = cfg['train']['optimizer'].get('wc', 1e-4)
    lr_patience  = cfg['train']['lr_scheduler'].get('patience', 20)

    optimizer = optim.AdamW([
        {'params': bert_params,  'lr': bert_lr},
        {'params': other_params, 'lr': head_lr},
    ], weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_patience,
        gamma=0.1,
    )

    # ---- training loop (mirrors train_CREMAD.py) ------------------------
    best_acc = 0.0
    # epoch_dict is an int in crema.json; guard against template's dict form
    _ed  = cfg['train']['epoch_dict']
    epochs = _ed if isinstance(_ed, int) else _ed.get('train_image_text', 50)
    check    = max(1, int(epochs / 10))

    for epoch in range(epochs):
        print(f"Epoch {epoch} is pending...")
        scheduler.step()

        model, ratio_a, loss_ave, loss_t_ave, loss_i_ave = train_text_image(
            epoch, train_loader, model, optimizer, args.merge_alpha
        )
        f1, acc, mAP, f1_t, acc_t, mAP_t, f1_i, acc_i, mAP_i = val(epoch, val_loader, model, args.merge_alpha)
        
        log_writer.writerow([
            epoch,
            loss_ave,
            loss_t_ave,
            loss_i_ave,
            f1,
            acc,
            mAP,
            f1_t,
            acc_t,
            mAP_t,
            f1_i,
            acc_i,
            mAP_i,
            ratio_a
        ])
        log_file.flush()
        print(f"ratio: {ratio_a:.3f}")

        if acc >= best_acc:
            best_acc = acc
            print('Find a better model and save it!')
            torch.save(
                model.state_dict(),
                f"{model_folder_path}/twitter_GB_best_model_{best_acc:.4f}.pth"
            )
        if (epoch + 1) % check == 0 or epoch == 0:
            print(f"ratio: {ratio_a:.3f}")
            if ratio_a > args.lam + 0.01:
                print("add_layer_image")
                model.add_layer(is_a=False)
            elif ratio_a < args.lam - 0.01:
                print("add_layer_text")
                model.add_layer(is_a=True)
