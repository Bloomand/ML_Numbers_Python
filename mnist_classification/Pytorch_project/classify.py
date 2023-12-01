from sklearn.metrics import f1_score
from sklearn.datasets import load_iris
import argparse
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from dict_unicls import get_data_from
from inspect import signature
import numpy as np
import os
import sklearn
from sklearn import svm
import pickle
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', choices=['train', 'test'])
    parser.add_argument('--datafile', default=r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\test.csv',
                        help='Path to csv file with samples.')
    parser.add_argument('--model_path', default=r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\model.pkl')

    print(parser.parse_args())
    return parser


class LinearSVM:
    def __init__(self, loading_file=None):
        self.classifier = None
        if loading_file is not None:
            self.classifier = self.load_model(loading_file)
        else:
            self.classifier = svm.LinearSVC()

    def load_model(self, path1):
        with open(path1, 'rb') as f:
            self.classifier = pickle.load(f)
        return self.classifier

    def save_model(self, path2):
        with open(path2, 'wb') as f:
            pickle.dump(self.classifier, f)

    def train(self, X_train, y_train):
        return self.classifier.fit(X_train, y_train)

    def inference(self, X_test):
        return self.classifier.predict(X_test)


if __name__ == '__main__':

    parser = get_args()
    args = parser.parse_args()
    if args.mode == 'train':
        cls = LinearSVM()
        X_train, Y_train, _ = get_data_from(args.datafile)
        cls.train(X_train.reshape(X_train.shape[0], -1), Y_train)
        cls.save_model(args.model_path)
    else:
        cls = LinearSVM(loading_file=args.model_path)

        X_test, Y_test, _ = get_data_from(args.datafile)
        y_pred = cls.inference(X_test.reshape(X_test.shape[0], -1))
        y_pred = np.array(y_pred, dtype=np.int32)

        with open('txt1.txt', 'w') as f:
            [print(x, file=f, end=' ') for x in y_pred]

        y_pred = np.array(y_pred, dtype=np.int32)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

        print(sklearn.metrics.accuracy_score(Y_test, y_pred))
    #X_train, y_train, _ = get_data_from('train1.txt', 'train')

    #first.train(X_train.reshape(X_train.shape[0], -1), y_train)
    #first.save_model('savetxt.txt')

    #print(f1_score(gt_labels, y_pred, average=None))
