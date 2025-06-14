import os
import time
import torch
import logging
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

import random


def cal_metrics(act_label, pred_lable):
    if not (hasattr(act_label, 'detach') and hasattr(pred_lable, 'detach')):
        raise ValueError("Inputs should be PyTorch tensors.")

    #print(act_label.shape, pred_lable.shape)
    act_label = torch.nan_to_num(act_label, nan=0.0)
    pred_lable = torch.nan_to_num(pred_lable, nan=0.0)

    auc = roc_auc_score(act_label.detach().cpu().numpy(), pred_lable.detach().cpu().numpy())
    precision, recall, _ = metrics.precision_recall_curve(act_label.detach().cpu().numpy(), pred_lable.detach().cpu().numpy())
    pr_auc = metrics.auc(recall, precision)

    pred_lables_binary = (pred_lable.detach().cpu() > 0.55).numpy()
    accuracy = accuracy_score(act_label.detach().cpu().numpy(), pred_lables_binary)
    precision_score_val = precision_score(act_label.detach().cpu().numpy(), pred_lables_binary, zero_division=1)
    recall_score_val = recall_score(act_label.detach().cpu().numpy(), pred_lables_binary)
    f1_score_val = f1_score(act_label.detach().cpu().numpy(), pred_lables_binary)

    return auc,  pr_auc, f1_score_val,recall_score_val, precision_score_val, accuracy



def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dir(args):


    cache_dir = os.path.join(args.res_dir, 'cache')

    check_dir(cache_dir)

    model_dir = os.path.join(args.res_dir, 'model')
    check_dir(model_dir)

    log_dir = os.path.join(args.res_dir, 'log')
    check_dir(log_dir)

    return cache_dir, model_dir, log_dir


def init_logger(log_dir):


    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s   %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_dir, time.strftime("%Y_%m_%d") + '.log'),
                        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging

def set_seed(fix_seed):
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(fix_seed)