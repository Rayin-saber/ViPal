# -*- coding: utf-8 -*-
'''
Created on Thur Oct 22 13:05:23 2020

@author: Luo Zihan
'''

import os
import numpy as np
import torch
import torchvision.models as models
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from method import reshape_to_linear
from method import train_test_split_data_pr
from method import train_test_split_data
from method import knn_cross_validation
from method import svm_cross_validation
from method import bayes_cross_validation
from method import logistic_cross_validation
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
    while True:
        print("Please chooe whether use prior knowledge:")
        prior_flag = input()
        # Methods without prior knowledge
        if prior_flag == "No":
            METHODS = ['TL', 'CNN']
            NETS = ['AlexNet', 'ResNet50', 'SqueezeNet', 'VGG']

            METHOD = METHODS[1]
            NET = NETS[0]

            flu_feature = np.loadtxt("flu_feature.csv", delimiter=",")
            flu_label = np.loadtxt("label.csv", delimiter=",", usecols=range(1, 2))

            scaler = preprocessing.StandardScaler()  # normalization
            flu_feature = scaler.fit_transform(flu_feature)

            # tranditional machine learning
            if METHOD == 'TL':
                train_x, test_x, train_y, test_y = train_test_split_data(flu_feature, flu_label, 0.2, SHUFFLE=True)
                knn_cross_validation(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
                svm_cross_validation(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
                bayes_cross_validation(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
                logistic_cross_validation(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)

            # convolutional nerual network
            elif METHOD == 'CNN':
                parameters = {
                    # Note, no learning rate decay implemented
                    'learning_rate': 0.0001,  # 0.0005,

                    # Size of mini batch
                    'batch_size': 4,

                    # Number of training iterations
                    'num_of_epochs': 200
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

                train_results = []
                test_results = []
                train_features, val_features, train_labels, val_labels = train_test_split_data(flu_feature, flu_label, 0.2, SHUFFLE=True)
                stratified_folder = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)  # cross-validation
                best_acc = 0
                for idx, (train_index, test_index) in enumerate(stratified_folder.split(train_features, train_labels)):
                    print(idx+1, "th folder...")
                    train_x = np.array(train_features)[train_index]
                    test_x = np.array(train_features)[test_index]
                    train_y = np.array(train_labels)[train_index]
                    test_y = np.array(train_labels)[test_index]

                    # reshape 147->21*7
                    train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, 21, 7))
                    test_x = np.reshape(test_x, (np.array(test_x).shape[0], 1, 21, 7))

                    train_x = torch.tensor(train_x, dtype=torch.float32)
                    train_y = torch.tensor(train_y, dtype=torch.int64)
                    test_x = torch.tensor(test_x, dtype=torch.float32)
                    test_y = torch.tensor(test_y, dtype=torch.int64)

                    # using GPU...
                    if torch.cuda.is_available():
                        train_x = train_x.cuda()
                        train_y = train_y.cuda()
                        test_x = test_x.cuda()
                        test_y = test_y.cuda()

                    train_result, test_result, best_acc = train_CNN_npr.train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y, False, best_acc)
                    train_results.append(train_result)
                    test_results.append(test_result)

                # calculate the average results
                T_results = np.array(train_results).mean(axis=0)
                V_results = np.array(test_results).mean(axis=0)
                print("##########################################################################")
                print("After 5 folders, the results are:")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tAUC %.3f' % (T_results[0], T_results[1], T_results[2], T_results[3], T_results[4]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tAUC %.3f' % (V_results[0], V_results[1], V_results[2], V_results[3], T_results[4]))

                # validation
                if NET == 'ResNet50':
                    model = ResNet(models.resnet50())
                elif NET == 'AlexNet':
                    model = AlexNet(models.alexnet())
                elif NET == 'SqueezeNet':
                    model = SqueezeNet(models.squeezenet1_0(), prior=False)
                elif NET == 'VGG':
                    model = VGG(models.vgg16(), prior=False)
                val_x = np.reshape(val_features, (np.array(val_features).shape[0], 1, 21, 7))
                val_x = torch.tensor(val_x, dtype=torch.float32)
                val_y = torch.tensor(val_labels, dtype=torch.int64)
                model.load_state_dict(torch.load('cnn_no_pr.pkl'))
                model.eval()
                val_scores = model(val_x)
                prediction = train_CNN_npr.predictions_from_output(val_scores)
                pred_prob = train_CNN_npr.calculate_prob(val_scores)
                fpr_cnn, tpr_cnn, _ = roc_curve(val_y.cpu().detach().numpy(), pred_prob.cpu().detach().numpy())
                AUC = auc(fpr_cnn, tpr_cnn)
                prediction = prediction.view_as(val_y)
                precision, recall, fscore, mcc, val_acc = evaluate(val_y, prediction)

                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tAUC %.3f' % (val_acc, precision, recall, fscore, AUC))
            else:
                raise NotImplementedError

            break

        # Methods with prior knowledge
        elif prior_flag == 'Yes':

            METHODS = ['TL', 'CNN']
            NETS = ['AlexNet', 'ResNet50', 'SqueezeNet', 'VGG']

            METHOD = METHODS[1]
            NET = NETS[0]

            flu_feature = np.loadtxt("flu_feature_mutation.csv", delimiter=",")
            flu_label = np.loadtxt("label.csv", delimiter=",", usecols=range(1, 2))

            scaler = preprocessing.StandardScaler()
            flu_feature = scaler.fit_transform(flu_feature)
            mutation_site = flu_feature[-22, -1]
            reassortment = flu_feature[-1]

            if METHOD == 'TL':
                train_x, test_x, train_y, test_y, train_mutation_site, test_mutation_site, train_reassortment, test_reassortment = train_test_split_data_pr(flu_feature, flu_label, 0.2, SHUFFLE=True)
                knn_cross_validation(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
                svm_cross_validation(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
                bayes_cross_validation(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
                logistic_cross_validation(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)

            elif METHOD == 'CNN':

                parameters = {
                    # Note, no learning rate decay implemented
                    'learning_rate': 0.002,  # 0.0005,

                    # Size of mini batch
                    'batch_size': 4,

                    # Number of training iterations
                    'num_of_epochs': 80
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

                train_results = []
                test_results = []
                train_features, val_features, train_labels, val_labels, train_mutation_site, val_mutation_site, train_reassortment, val_reassortment = train_test_split_data_pr(flu_feature, flu_label, 0.2, SHUFFLE=True)
                # print(train_mutation_site.shape) (390, 21)
                stratified_folder = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)  # cross-validation
                best_acc = 0
                for idx, (train_index, test_index) in enumerate(stratified_folder.split(train_features, train_labels)):
                    print(idx+1, "th folder...")
                    train_x = np.array(train_features)[train_index]
                    test_x = np.array(train_features)[test_index]
                    train_y = np.array(train_labels)[train_index]
                    test_y = np.array(train_labels)[test_index]

                    # print(train_x[:, -22:-1].shape) (312,21)
                    train_mutation_site = torch.tensor(np.repeat(train_x[:, -22:-1], 2, axis=1), dtype=torch.float32)
                    train_reassortment = torch.tensor(np.repeat(train_x[:, -1].reshape(train_x[:, -1].shape[0], 1), 2, axis=1), dtype=torch.float32)
                    test_mutation_site = torch.tensor(np.repeat(test_x[:, -22:-1], 2, axis=1), dtype=torch.float32)
                    test_reassortment = torch.tensor(np.repeat(test_x[:, -1].reshape(test_x[:, -1].shape[0], 1), 2, axis=1), dtype=torch.float32)

                    # reshape 169->13*13
                    train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, 13, 13))  # the dimension is 169
                    test_x = np.reshape(test_x, (np.array(test_x).shape[0], 1, 13, 13))
                    train_x = torch.tensor(train_x, dtype=torch.float32)
                    train_y = torch.tensor(train_y, dtype=torch.int64)
                    test_x = torch.tensor(test_x, dtype=torch.float32)
                    test_y = torch.tensor(test_y, dtype=torch.int64)

                    # using GPU...
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
                    
                    train_results.append(train_result)
                    test_results.append(test_result)

                # calculate the average results
                T_results = np.array(train_results).mean(axis=0)
                V_results = np.array(test_results).mean(axis=0)
                print("##############################################################N############")
                print("After 5 folders, the results are:")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tAUC %.3f' % (T_results[0], T_results[1], T_results[2], T_results[3], T_results[4]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tAUC %.3f' % (V_results[0], V_results[1], V_results[2], V_results[3], T_results[4]))

                # validation
                if NET == 'ResNet50':
                    model = ResNet(models.resnet50())
                elif NET == 'AlexNet':
                    model = AlexNet(models.alexnet())
                elif NET == 'SqueezeNet':
                    model = SqueezeNet(models.squeezenet1_0(), prior=True)
                elif NET == 'VGG':
                    model = VGG(models.vgg16(), prior=True)
                if torch.cuda.is_available():
                    model.cuda()
                
                PR_model = PR(2, 22)

                val_x = np.reshape(val_features, (np.array(val_features).shape[0], 1, 13, 13))
                val_x = torch.tensor(val_x, dtype=torch.float32)
                val_y = torch.tensor(val_labels, dtype=torch.int64)

                val_mutation_site = torch.tensor(val_mutation_site, dtype=torch.float32)
                val_mutation_site = torch.tensor(np.repeat(val_mutation_site, 2, axis=1), dtype=torch.float32)
                val_reassortment = torch.tensor(val_reassortment, dtype=torch.float32)
                val_reassortment = torch.tensor(np.repeat(val_reassortment.reshape(val_reassortment.shape[0], 1), 2, axis=1), dtype=torch.float32)

                if torch.cuda.is_available():
                    val_x = val_x.cuda()
                    val_y = val_labels.cuda()
                    val_mutation_site = val_mutation_site.cuda()
                    val_reassortment = val_reassortment.cuda()

                model.load_state_dict(torch.load('cnn_pr.pkl'))
                PR_model.load_state_dict(torch.load('pr.pkl'))
                
                model.eval()
                val_scores = model(val_x)
                
                val_desired = PR_model(val_mutation_site, val_reassortment)

                predictions = train_CNN_pr.predictions_from_output(val_scores + val_desired)
                predictions = predictions.view_as(val_y)

                precision, recall, fscore, mcc, val_acc = evaluate(val_y, predictions)
                pred_prob = train_CNN_pr.calculate_prob(val_scores + val_desired)
                fpr_cnn, tpr_cnn, _ = roc_curve(val_y.cpu().detach().numpy(), pred_prob.cpu().detach().numpy())
                AUC = auc(fpr_cnn, tpr_cnn)

                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tAUC %.3f' % (val_acc, precision, recall, fscore, AUC))

            else:
                raise NotImplementedError

            break
        else:
            print("Input error! Please try again.")
            continue
