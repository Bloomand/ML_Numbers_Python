# -*- coding: utf-8 -*-
import os
import time
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', action='store_true')


def create_train(folder):
    fds = os.listdir(folder)
    data = list()

    for fname in fds:
        label = fname[-5]
        data.append([r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\train\\'+fname, label])

    dt = pd.DataFrame(data, columns =['fname', 'label'])
    dt.to_csv('train.csv', index=False, sep=',')


def create_test(folder):
    fds = os.listdir(folder)
    data = list()

    for fname in fds:
        label = fname[-5]
        data.append([r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\test\\'+fname, label])

    dt = pd.DataFrame(data, columns =['fname', 'label'])
    dt.to_csv('test.csv', index=False, sep=',')


def get_data_from(csv):
    data_from_csv = pd.read_csv(csv).to_numpy()
    H = 28
    W = 28
    im_arr = np.zeros((len(data_from_csv), H, W))

    label_arr = np.array(data_from_csv[:, 1], dtype=np.int32)
    paths_arr = data_from_csv[:, 0]

    for ind, f in enumerate(paths_arr):
        im_arr[ind] = plt.imread(f)

    return im_arr, label_arr, paths_arr


if __name__ == '__main__':
    create_train(r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\train')
    create_test(r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\test')
