#!/usr/bin/python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import random
import wget
import tarfile
import os

url_train = "https://pjreddie.com/media/files/mnist_train.tar.gz"
file_train = "mnist_train.tar.gz"

url_test = "https://pjreddie.com/media/files/mnist_test.tar.gz"
file_test = "mnist_test.tar.gz"

def download_unzip(url, fname):
    wget.download(url, fname)
    iFile = tarfile.open(fname, "r:gz")
    iFile.extractall()
    iFile.close()

def print_images():
    path = os.listdir('train')
    randomimage = np.random.choice(path)
    path2 = plt.imread('train/' + randomimage)
    plt.imshow(path2, cmap='gray')
    plt.show()

    # 2
    x = random.randint(1, 10)
    y = random.randint(1, 10)
    print('x: ' + str(x) + '\ny: ' + str(y))
    G = gs.GridSpec(y, x)
    for i in range(1, x * y + 1):
        randomimage = np.random.choice(path)
        plt.subplot(y, x, i).imshow(plt.imread('train/' + randomimage), cmap='gray')
    # plt.savefig('something.png')
    plt.show()

def createParser():
    parser=argparse.ArgumentParser()
    parser.add_argument('name',nargs='?')
    return parser
def main():
    pass

if __name__ == "__main__":
    parser=createParser()
    namespace=parser.parse_args()
    if namespace.name:
        print('hello,{}!'.format(namespace.name))
    else:
        print('Hello,world!')