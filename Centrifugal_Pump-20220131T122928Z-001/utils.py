# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def create_dirs(dirs):
    """
    Create dirs. (recurrent)
    :param dirs: a list directory path.
    :return: None
    """
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=False)

def write2txt(content, file_path):
    """
    Write array to .txt file.
    :param content: array.
    :param file_path: destination file path.
    :return: None.
    """
    try:
        file_name = file_path.split('/')[-1]
        dir_path = file_path.replace(file_name, '')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path, 'w+') as f:
            for item in content:
                f.write(' '.join([str(i) for i in item]) + '\n')

        print("write over!")
    except IOError:
        print("fail to open file!")

def write2csv(content, file_path):
    """
    Write array to .csv file.
    :param content: array.
    :param file_path: destination file path.
    :return: None.
    """
    try:
        temp = file_path.split('/')[-1]
        temp = file_path.replace(temp, '')
        if not os.path.exists(temp):
            os.makedirs(temp)

        with open(file_path, 'w+', newline='') as f:
            csv_writer = csv.writer(f, dialect='excel')
            for item in content:
                csv_writer.writerow(item)

        print("write over!")
    except IOError:
        print("fail to open file!")

def calculate_auroc(predictions, labels):
    if np.max(labels) ==1 and np.min(labels)==0:
        fpr_list, tpr_list, _ = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=True)
        auroc = metrics.roc_auc_score(labels, predictions)
    else:
        fpr_list, tpr_list = [], []
        auroc = np.nan

    return fpr_list, tpr_list, auroc

def calculate_aupr(predictions, labels):
    if np.max(labels) == 1 and np.min(labels) == 0:
        precision_list, recall_list, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
        aupr = metrics.auc(recall_list, precision_list)
    else:
        precision_list, recall_list = [], []
        aupr = np.nan
    return precision_list, recall_list, aupr

def plot_loss_curve(epoch, train_loss, val_loss, file_path):
   
    plt.figure()
    plt.plot(epoch, train_loss, lw=1, label = 'Train Loss')
    plt.plot(epoch, val_loss, lw=1, label = 'Valid Loss')
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.savefig(file_path)


def plot_roc_curve(fpr_list, tpr_list, file_path):
   
    plt.figure()
    for i in range(0, 1):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"normal (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_normal.jpg'))

    plt.figure()
    
    for i in range(1,2):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"impeller_wearing (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_impeller_wearing.jpg'))

    plt.figure()
    for i in range(2,3):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"bearing_roller (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_bearing_roller.jpg'))
    
    plt.figure()
    for i in range(3,4):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"outer_race (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_outer_race.jpg'))
    
    plt.figure()
    for i in range(4,5):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"inner_race (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_inner_race.jpg'))


def plot_pr_curve(precision_list, recall_list, file_path):

    plt.figure()
    for i in range(0,1):
        plt.plot(precision_list[i], recall_list[i], lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"normal (PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_normal.jpg'))

    plt.figure()
 
    for i in range(1,2):
        plt.plot(recall_list[i], precision_list[i], lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"impeller_wearing (PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_impeller_wearing.jpg'))

    plt.figure()
    for i in range(2, 3):
        plt.plot(precision_list[i], recall_list[i], lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"bearing_roller(PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_bearing_roller.jpg'))
    
    plt.figure()
    for i in range(3, 4):
        plt.plot(precision_list[i], recall_list[i], lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"outer_race(PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_outer_race.jpg'))
    
    plt.figure()
    for i in range(4, 5):
        plt.plot(precision_list[i], recall_list[i], lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"inner_race (PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_inner_race .jpg'))