# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras

from model import lenet_kh
from trainer import Trainer
from loader import get_train_data, get_valid_data, get_test_data
from utils import plot_loss_curve, plot_roc_curve, plot_pr_curve
from utils import calculate_auroc, calculate_aupr
from utils import create_dirs, write2txt, write2csv

np.random.seed(0)
tf.random.set_seed(0)

def train():
    dataset_train = get_train_data(50) #batch size for training =50
    dataset_valid = get_valid_data(50) #batch size for validation =50

    model = lenet_kh()
    loss_object = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam()
    trainer = Trainer(
        model=model,
        loss_object=loss_object,
        optimizer=optimizer,
        experiment_dir='./result/lenet_kh')

    history = trainer.train(dataset_train, dataset_valid, epoch=52, train_steps=int(np.ceil(50000 / 50)),
                  valid_steps=int(np.ceil(500 / 50)), dis_show_bar=True)


    # Plot the loss curve of training and validation, and save the loss value of training and validation.
    print('\n history dict: ', history)
    epoch = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['valid_loss']
    plot_loss_curve(epoch, train_loss, val_loss, './result/lenet_kh/model_loss.jpg')
    np.savez('./result/lenet_kh/model_loss.npz', train_loss = train_loss, val_loss = val_loss)


def test():
    dataset_test = get_test_data(50) #batch size for test =50

    model = lenet_kh()
    loss_object = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam()
    trainer = Trainer(
        model=model,
        loss_object=loss_object,
        optimizer=optimizer,
        experiment_dir='./result/lenet_kh')

    result, label = trainer.test(dataset_test, test_steps=int(np.ceil(10000 / 50)), dis_show_bar=True)

    #result = np.mean((result[0:650], result[650:]), axis=0)
    
    result_shape = np.shape(result)
    #label = label[0:4639]

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in tqdm(range(result_shape[1]), ascii=True):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], label[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], label[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    plot_roc_curve(fpr_list, tpr_list, './result/lenet_kh/')
    plot_pr_curve(precision_list, recall_list, './result/lenet_kh/')

    header = np.array([['auroc', 'aupr']])
    content = np.stack((auroc_list, aupr_list), axis=1)
    content = np.concatenate((header, content), axis=0)
    write2csv(content, './result/lenet_kh/result.csv')
    write2txt(content, './result/lenet_kh/result.txt')
    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    print('AVG-AUROC:{:.3f}, AVG-AUPR:{:.3f}.\n'.format(avg_auroc, avg_aupr))

if __name__ == '__main__':
    # Parses the command line arguments and returns as a simple namespace.
    parser = argparse.ArgumentParser(description='main_lenet_kh.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.')
    args = parser.parse_args()

    # Selecting the execution mode (keras).
    create_dirs(['./result/lenet_kh/'])
    if args.exe_mode == 'train':
        train()
        print("finish_train")
    elif args.exe_mode == 'test':
        test()