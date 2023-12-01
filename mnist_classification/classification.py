# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
def create_train(folder):
    fds = os.listdir(folder)
    data = list()
    for img in fds:
        fname = img
        index, label = os.path.splitext(img)[0].split('-num')
        index = int(index)
        data.append([index, fname, label])
    data.sort()
    dt=pd.DataFrame(data, columns =['index', 'fname', 'label'])
    dt.to_csv('train.csv', index=False, sep=',')
def create_test(folder):
    fds = os.listdir(folder)
    data=list()
    for img in fds:   #проход по папке
        fname = img
        index = os.path.splitext(img)[0].split('-num')[0]
        index = int(index)
        data.append([index, fname])
    data.sort()
    dt = pd.DataFrame(data,columns=['index', 'fname'])
    dt.to_csv('test.csv', index=False, sep=',')
def get_data_from (txt, mode):
    date_from_txt = np.genfromtxt(txt, dtype=str)
    xy=[]
    for f in date_from_txt:
        x = (f.split(','))[1]
        if mode!= 'test':
            y = (f.split(','))[2]
            xy.append([str(x), str(y)])
        else:
            xy.append([str(x)])
    return xy
def sublet():
    pass
def show(xy):
    path = os.listdir('test')
    rimage = np.random.choice(path)
    path2 = plt.imread('test/' + rimage)
    rimage = np.random.choice(path)
    path3 = plt.imread('test/' + rimage)
    rimage = np.random.choice(path)

    path4 = plt.imread('test/' + rimage)

    fig, axes = plt.subplots(1, 3)
    path2[:, 0] = 0
    axes[0].imshow(path2)
    path3[:, 1] = 0
    axes[1].imshow(path3)

    path4[:, 2]
    axes[2].imshow(path4)

    for ax in axes:
        ax.set_yticks([])

    fig.set_figwidth(12)
    fig.set_figheight(6)

    plt.show()


if __name__ == '__main__':
    create_train(r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\train')
    create_test(r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\test')
    xy = (get_data_from('test.csv', 'test'))
    print(xy)
    show(xy)
