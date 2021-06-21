'''
Created on Wed Oct 29 15:45:03 2020

@author: Rui Yin and Luo Zihan
'''

import os
import numpy as np
import torch
import torchvision.models as models
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from method import reshape_to_linear
from method import train_test_split_data_pr
from method import train_test_split_data
from method import AlexNet
from method import ResNet
from method import SqueezeNet
from method import VGG
from method import PR
from validation_ISMB import evaluate
import train_CNN_npr
import train_CNN_pr


if __name__ == "__main__":
    
    os.chdir('/Users/lzh/Desktop/NTU/Prior Knowledge/Prior knowledge with CTD/github/data')
    SUBTYPE = ['H1N1', 'H3N2', 'H5N1', 'other']
    while True:

        print("Please chooe whether use prior knowledge:")
        prior_flag = input()
        print("Please chooe subtype:")
        subtype_name = input()
        if subtype_name not in SUBTYPE:
            print("Not Implemented Error! Subtype should be 'H1N1', 'H3N2', 'H5N1' or 'other'.")
            break
        
        # Methods without prior knowledge
        if prior_flag == "No":
            METHOD = 'CNN'
            NETS = ['AlexNet', 'ResNet50', 'SqueezeNet', 'VGG']

            NET = NETS[2]
            
            flu_feature = np.loadtxt("flu_feature_"+subtype_name+".csv", delimiter=",")
            flu_label = np.loadtxt("label_"+subtype_name+".csv", delimiter=",", usecols=range(1, 2))

            scaler = preprocessing.StandardScaler()  # normalization
            flu_feature = scaler.fit_transform(flu_feature)

            # convolutional nerual network
            if METHOD == 'CNN':
                parameters = {
                    # Note, no learning rate decay implemented
                    'learning_rate': 0.001,  # 0.0005,

                    # Size of mini batch
                    'batch_size': 4,

                    # Number of training iterations
                    'num_of_epochs': 300
                }

                if NET == 'ResNet50':
                    net = ResNet(models.resnet50(pretrained=True))
                    print("Using ResNet50...")
                elif NET == 'AlexNet':
                    net = AlexNet(models.alexnet(pretrained=True))
                    print("Using AlexNet...")
                elif NET == 'SqueezeNet':
                    net = SqueezeNet(models.squeezenet1_0(pretrained=True), prior=False)
                    print("Using SqueezeNet...")
                elif NET == 'VGG':
                    net = VGG(models.vgg16(pretrained=True), prior=False)
                    print("Using VGG-16...")
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()

                train_features, val_features, train_labels, val_labels = train_test_split_data(flu_feature, flu_label, 0.2, SHUFFLE=True)
                best_acc = 0
                
                train_x = np.array(train_features)
                train_y = np.array(train_labels)
                test_x = np.array(val_features)
                test_y = np.array(val_labels)

                # reshape 147->21*7
                train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, 21, 7))
                test_x = np.reshape(test_x, (np.array(test_x).shape[0], 1, 21, 7))

                # using GPU...
                train_x = torch.tensor(train_x, dtype=torch.float32)
                train_y = torch.tensor(train_y, dtype=torch.int64)
                test_x = torch.tensor(test_x, dtype=torch.float32)
                test_y = torch.tensor(test_y, dtype=torch.int64)
                
                if torch.cuda.is_available():
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()

                train_result, test_result, best_acc = train_CNN_npr.train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y, False, best_acc)
            else:
                raise NotImplementedError

            break

        # Methods with prior knowledge
        elif prior_flag == 'Yes':

            METHOD = 'CNN'
            NETS = ['AlexNet', 'ResNet50', 'SqueezeNet', 'VGG']

            NET = NETS[2]

            # print("H3N2")
            flu_feature = np.loadtxt("flu_feature_mutation_"+subtype_name+".csv", delimiter=",")
            flu_label = np.loadtxt("label_"+subtype_name+".csv", delimiter=",", usecols=range(1, 2))

            scaler = preprocessing.StandardScaler()
            flu_feature = scaler.fit_transform(flu_feature)
            mutation_site = flu_feature[-22, -1]
            reassortment = flu_feature[-1]

            if METHOD == 'CNN':

                parameters = {
                    # Note, no learning rate decay implemented
                    'learning_rate': 0.002,  # 0.0005,

                    # Size of mini batch
                    'batch_size': 4,

                    # Number of training iterations
                    'num_of_epochs': 300
                }

                if NET == 'ResNet50':
                    net = ResNet(models.resnet50(pretrained=True))
                    print("Using ResNet50...")
                elif NET == 'AlexNet':
                    net = AlexNet(models.alexnet(pretrained=True))
                    print("Using AlexNet...")
                elif NET == 'SqueezeNet':
                    net = SqueezeNet(models.squeezenet1_0(pretrained=True), prior=True)
                    print("Using SqueezeNet...")
                elif NET == 'VGG':
                    net = VGG(models.vgg16(pretrained=True), prior=True)
                    print("Using VGG-16...")
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()

                train_features, val_features, train_labels, val_labels = train_test_split_data(flu_feature, flu_label, 0.2, SHUFFLE=True)
                best_acc = 0
                
                train_x = np.array(train_features)
                train_y = np.array(train_labels)
                test_x = np.array(val_features)
                test_y = np.array(val_labels)

                # print(train_x[:, -22:-1].shape) (312,21)
                train_mutation_site = torch.tensor(np.repeat(train_x[:, -22:-1], 2, axis=1), dtype=torch.float32)
                train_reassortment = torch.tensor(np.repeat(train_x[:, -1].reshape(train_x[:, -1].shape[0], 1), 2, axis=1), dtype=torch.float32)
                test_mutation_site = torch.tensor(np.repeat(test_x[:, -22:-1], 2, axis=1), dtype=torch.float32)
                test_reassortment = torch.tensor(np.repeat(test_x[:, -1].reshape(test_x[:, -1].shape[0], 1), 2, axis=1), dtype=torch.float32)

                # reshape 169->13*13
                train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, 13, 13))  # the dimension is 169
                test_x = np.reshape(test_x, (np.array(test_x).shape[0], 1, 13, 13))

                # using GPU...
                train_x = torch.tensor(train_x, dtype=torch.float32)
                train_y = torch.tensor(train_y, dtype=torch.int64)
                test_x = torch.tensor(test_x, dtype=torch.float32)
                test_y = torch.tensor(test_y, dtype=torch.int64)

                if torch.cuda.is_available():
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                    train_mutation_site = train_mutation_site.cuda()
                    test_mutation_site = test_mutation_site.cuda()
                    train_reassortment = train_reassortment.cuda()
                    test_reassortment = test_reassortment.cuda()

                pr = PR(2, 22)
                
                train_result, test_result, best_acc = train_CNN_pr.train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y, train_mutation_site, train_reassortment, test_mutation_site, test_reassortment, pr, False, best_acc)
                
            else:
                raise NotImplementedError

            break
        else:
            print("Input error! Please try again.")
            continue
