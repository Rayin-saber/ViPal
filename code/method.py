# -*- coding: utf-8 -*-
"""
Created on Thur Oct 22 09:20:17 2020

@author: Rui Yin and Luo Zihan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import math
import torch
from sklearn import neighbors
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.preprocessing import label_binarize
from validation_ISMB import evaluate
from torch import nn
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reshape_to_linear(x):
    output = np.reshape(x, (x.shape[0], -1))

    return output


def train_test_split_data_pr(feature, label, split_ratio, SHUFFLE):
    setup_seed(50)
    train_x, test_x, train_y, test_y, train_mutation_site, test_mutation_site, train_reassortment, test_reassortment = [], [], [], [], [], [], [], []
    feature_new, label_new = [], []
    num_of_training = int(math.floor(len(feature) * (1 - split_ratio)))
    shuffled_index = np.arange(len(feature))
    if SHUFFLE is True:
        random.shuffle(shuffled_index)
    else:
        pass
    for i in range(0, len(feature)):
        feature_new.append(feature[shuffled_index[i]])
        label_new.append(label[shuffled_index[i]])
    feature_new = np.array(feature_new)
    label_new = np.array(label_new)

    train_x = feature_new[:num_of_training]
    train_y = label_new[:num_of_training]
    test_x = feature_new[num_of_training:]
    test_y = label_new[num_of_training:]
    train_mutation_site = feature_new[:num_of_training, -22:-1]
    test_mutation_site = feature_new[num_of_training:, -22:-1]
    train_reassortment = feature_new[:num_of_training, -1:]
    test_reassortment = feature_new[num_of_training:, -1:]

    return train_x, test_x, train_y, test_y, train_mutation_site, test_mutation_site, train_reassortment, test_reassortment


def train_test_split_data(feature, label, split_ratio, SHUFFLE):
    setup_seed(11)
    train_x, test_x, train_y, test_y = [], [], [], []
    feature_new, label_new = [], []
    num_of_training = int(math.floor(len(feature) * (1 - split_ratio)))
    shuffled_index = np.arange(len(feature))
    if SHUFFLE is True:
        random.shuffle(shuffled_index)
    else:
        pass
    for i in range(0, len(feature)):
        feature_new.append(feature[shuffled_index[i]])
        label_new.append(label[shuffled_index[i]])

    train_x = feature_new[:num_of_training]
    train_y = label_new[:num_of_training]
    test_x = feature_new[num_of_training:]
    test_y = label_new[num_of_training:]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, test_x, train_y, test_y


def bayes_cross_validation(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = BernoulliNB()
    # calculate the accuracy
    cross_acc = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')

    # calculate the precision
    cross_pre = cross_val_score(clf, X, Y, cv=5, scoring='precision')

    # calculate the recall score
    cross_rec = cross_val_score(clf, X, Y, cv=5, scoring='recall')

    # calculate the f1 score
    cross_f1 = cross_val_score(clf, X, Y, cv=5, scoring='f1')

    clf = clf.fit(X, Y)
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)

    print("The result of Bayes is:")
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (cross_acc.mean(), cross_pre.mean(), cross_rec.mean(), cross_f1.mean()))
    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))


def knn_cross_validation(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = neighbors.KNeighborsClassifier()
    # calculate the accuracy
    cross_acc = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')

    # calculate the precision
    cross_pre = cross_val_score(clf, X, Y, cv=5, scoring='precision')

    # calculate the recall score
    cross_rec = cross_val_score(clf, X, Y, cv=5, scoring='recall')

    # calculate the f1 score
    cross_f1 = cross_val_score(clf, X, Y, cv=5, scoring='f1')

    clf = clf.fit(X, Y)
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)

    print("The result of KNN is:")
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (cross_acc.mean(), cross_pre.mean(), cross_rec.mean(), cross_f1.mean()))
    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))


def svm_cross_validation(X, Y, X_test, Y_test):
    setup_seed(20)
    clf = SVC(gamma='auto', class_weight='balanced', probability=True)
    # calculate the accuracy
    cross_acc = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')

    # calculate the precision
    cross_pre = cross_val_score(clf, X, Y, cv=5, scoring='precision')

    # calculate the recall score
    cross_rec = cross_val_score(clf, X, Y, cv=5, scoring='recall')

    # calculate the f1 score
    cross_f1 = cross_val_score(clf, X, Y, cv=5, scoring='f1')

    clf = clf.fit(X, Y)
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)

    print("The result of SVM is:")
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (cross_acc.mean(), cross_pre.mean(), cross_rec.mean(), cross_f1.mean()))
    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))


def logistic_cross_validation(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = linear_model.LogisticRegression()
    # calculate the accuracy
    cross_acc = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')

    # calculate the precision
    cross_pre = cross_val_score(clf, X, Y, cv=5, scoring='precision')

    # calculate the recall score
    cross_rec = cross_val_score(clf, X, Y, cv=5, scoring='recall')

    # calculate the f1 score
    cross_f1 = cross_val_score(clf, X, Y, cv=5, scoring='f1')

    clf = clf.fit(X, Y)
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)

    print("The result of Logistic Regression is:")
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (cross_acc.mean(), cross_pre.mean(), cross_rec.mean(), cross_f1.mean()))
    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))



class ResNet(nn.Module):
    def __init__(self, model):
        super(ResNet, self).__init__()
        # 取掉model的后1层
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer1 = nn.Linear(2048, 256)  # 加上一层参数修改好的全连接层
        self.Linear_layer2 = nn.Linear(256, 2)
        # self.Linear_layer3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        out = self.softmax(out)
        return out


class VGG(nn.Module):
    def __init__(self, model, prior):
        super(VGG, self).__init__()
        # 取掉model的后1层
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-2])
        if prior is False:
            self.Linear_layer1 = nn.Linear(2816, 256)
        else:
            self.Linear_layer1 = nn.Linear(3136, 256)
        # self.Linear_layer1 = nn.Linear(3136, 256)  # 加上一层参数修改好的全连接层2816
        self.Linear_layer2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        out = self.softmax(out)
        return out


class AlexNet(nn.Module):
    def __init__(self, model):
        super(AlexNet, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer1 = nn.Linear(2304, 256)
        self.Linear_layer2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        out = self.softmax(out)
        return out


class SqueezeNet(nn.Module):
    def __init__(self, model, prior):
        super(SqueezeNet, self).__init__()
        self.conv_layer = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.squeeze_layer = nn.Sequential(*list(model.children())[1:-1])
        if prior is False:
            self.Linear_layer1 = nn.Linear(1408, 256)
        else:
            self.Linear_layer1 = nn.Linear(1568, 256)
        self.Linear_layer2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.squeeze_layer(x)
        # x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        out = self.softmax(out)
        return out


class PR(nn.Module):

    def __init__(self, n_labels, n_features):
        super(PR, self).__init__()

        self.n_labels = n_labels
        self.n_features = n_features

        # gamma
        self.gamma = nn.Parameter(torch.FloatTensor(torch.randn(self.n_features, self.n_labels)))
        if torch.cuda.is_available():
            self.gamma = nn.Parameter(torch.FloatTensor(torch.randn(self.n_features, self.n_labels)).cuda())

    def forward(self, mutation_site, reassortment):
        # number of patients
        n_samples = reassortment.size()[0]

        # concatenation
        f = torch.cat((mutation_site, reassortment), 1)
        f = f.view(n_samples, self.n_features, self.n_labels)

        gamma = self.gamma.unsqueeze(0).expand(n_samples, self.n_features, self.n_labels)
        f = (f * gamma).sum(1)

        f = torch.exp(f)
        f_sum = torch.sum(f, 1).unsqueeze(1).expand(n_samples, self.n_labels)

        pr = f / f_sum

        return pr
