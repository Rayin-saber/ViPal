# -*- coding: utf-8 -*-
'''
Created on Mon Oct 19 20:25:03 2020

@author: Luo Zihan
'''

from __future__ import division

import torch
import torch.nn.functional as F
import math
import time
from validation_ISMB import get_confusion_matrix
from validation_ISMB import evaluate
from validation_ISMB import get_time_string
from sklearn.metrics import roc_curve, auc

feature_vectors = []


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def predictions_from_output(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    _, predictions = prob.topk(1)
    return predictions


def calculate_prob(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    pred_probe, _ = prob.topk(1)
    return pred_probe


def kl_loss(q, p, log_eps):
    log_q = torch.log(q + log_eps)
    log_p = torch.log(p + log_eps)
    # print(log_q.size(), log_p.size())
    log_diff = log_q - log_p
    kl = torch.sum(torch.sum(q * log_diff, 1), 0)
    # print(q.size()[0])
    return kl/q.size()[0]


def verify_model(model, X, Y, batch_size):
    """
    Checks the loss at initialization of the model and asserts that the
    training examples in a batch aren't mixed together by backpropagating.
    """
    print('Sanity checks:')
    criterion = torch.nn.CrossEntropyLoss()
    scores, _ = model(X, model.init_hidden(Y.shape[0]))
    print(' Loss @ init %.3f, expected ~%.3f' % (criterion(scores, Y).item(), -math.log(1 / model.output_dim)))

    mini_batch_X = X[:, :batch_size, :]
    mini_batch_X.requires_grad_()
    criterion = torch.nn.MSELoss()
    scores, _ = model(mini_batch_X, model.init_hidden(batch_size))

    non_zero_idx = 1
    perfect_scores = [[0, 0] for i in range(batch_size)]
    not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0] for i in range(batch_size)]

    scores.data = torch.FloatTensor(not_perfect_scores)
    Y_perfect = torch.FloatTensor(perfect_scores)
    loss = criterion(scores, Y_perfect)
    loss.backward()

    zero_tensor = torch.FloatTensor([0] * X.shape[2])
    for i in range(mini_batch_X.shape[0]):
        for j in range(mini_batch_X.shape[1]):
            if sum(mini_batch_X.grad[i, j] != zero_tensor):
                assert j == non_zero_idx, 'Input with loss set to zero has non-zero gradient.'

    mini_batch_X.detach()
    print('Backpropagated dependencies OK')


def train_cnn(model, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, mutation_site, reassortment,
              mutation_site_test, reassortment_test, pr, verify, best_acc):
    """
    Training loop for a model utilizing hidden states.

    verify enables sanity checks of the model.
    epochs decides the number of training iterations.
    learning rate decides how much the weights are updated each iteration.
    batch_size decides how many examples are in each mini batch.
    show_attention decides if attention weights are plotted.
    """
    Weight_Decay = 0.0001
    alpha = 1
    beta = 1
    log_eps = 1e-8
    print("alpha:", alpha, " beta:", beta)
    flag_pr = 3
    learning_rate_pr = 0.002
    Weight_Decay_pr = 0.001
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    op_title = 'SGD'

    optimizer_pr = torch.optim.Adadelta(pr.parameters(), rho=0.95,weight_decay=Weight_Decay_pr)
    op_pr_title = 'Adadelta'
    
    print("Optimizers:", op_title, " & ", op_pr_title, "\nLearning Rate:", learning_rate, " & ", learning_rate_pr)
    criterion = torch.nn.CrossEntropyLoss()
    num_of_examples = X.shape[0]
    num_of_batches = math.floor(num_of_examples / batch_size)

    # if verify:
    #    verify_model(model, X, Y, batch_size)

    all_losses = []
    all_val_losses = []
    all_accs = []
    all_pres = []
    all_recs = []
    all_fscores = []
    all_mccs = []
    all_val_accs = []

    best_loss = 10
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        running_acc = 0
        running_pre = 0
        running_pre_total = 0
        running_rec = 0
        running_rec_total = 0
        epoch_fscore = 0
        running_mcc_numerator = 0
        running_mcc_denominator = 0
        running_rec_total = 0
        if verify:
            hidden = model.init_hidden(batch_size)

        for count in range(0, num_of_examples - batch_size + 1, batch_size):
            if verify:
                hidden = repackage_hidden(hidden)
                Y_batch = Y[count:count + batch_size]
                X_batch = X[count:count + batch_size, :, :]
                # print(X_batch.shape, np.array(hidden).shape)
                scores, hidden = model(X_batch, hidden)
                # print(scores.shape)

            else:
                Y_batch = Y[count:count + batch_size]
                # X_batch = X[count:count + batch_size, :, :]
                X_batch = X[count:count + batch_size, :, :, :]
                scores = model(X_batch)
                # print(scores.shape)
            mutation_batch = mutation_site[count:count + batch_size]
            reassortment_batch = reassortment[count:count + batch_size]
            desired = pr(mutation_batch, reassortment_batch)
            logit_loss = criterion(scores, Y_batch)
            kl = kl_loss(desired, scores, log_eps)
            # print(desired, scores)
            desired_cross = criterion(desired, Y_batch)

            loss = logit_loss + alpha * kl + beta * desired_cross
            # print(logit_loss, kl, desired_cross)

            optimizer.zero_grad()
            optimizer_pr.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_pr.step()

            predictions = predictions_from_output(scores + desired)

            conf_matrix = get_confusion_matrix(Y_batch, predictions)
            TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
            running_acc += TP + TN
            running_pre += TP
            running_pre_total += TP + FP
            running_rec += TP
            running_rec_total += TP + FN
            running_mcc_numerator += (TP * TN - FP * FN)
            if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0:
                running_mcc_denominator += 0
            else:
                running_mcc_denominator += math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            running_loss += loss.item()
        elapsed_time = time.time() - start_time
        epoch_acc = running_acc / Y.shape[0]
        all_accs.append(epoch_acc)
        if running_pre_total == 0:
            epoch_pre = 0
        else:
            epoch_pre = running_pre / running_pre_total
        all_pres.append(epoch_pre)

        if running_rec_total == 0:
            epoch_rec = 0
        else:
            epoch_rec = running_rec / running_rec_total
        all_recs.append(epoch_rec)

        if (epoch_pre + epoch_rec) == 0:
            epoch_fscore = 0
        else:
            epoch_fscore = 2 * epoch_pre * epoch_rec / (epoch_pre + epoch_rec)
        all_fscores.append(epoch_fscore)

        if running_mcc_denominator == 0:
            epoch_mcc = 0
        else:
            epoch_mcc = running_mcc_numerator / running_mcc_denominator
        all_mccs.append(epoch_mcc)

        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        with torch.no_grad():
            model.eval()
            if verify:
                test_scores, _ = model(X_test, model.init_hidden(Y_test.shape[0]))
            else:
                test_scores = model(X_test)
            test_desired = pr(mutation_site_test, reassortment_test)
            # print(test_desired, "\n", test_scores)

            predictions = predictions_from_output(test_scores + test_desired)
            predictions = predictions.view_as(Y_test)
            pred_prob = calculate_prob(test_scores + test_desired)

            precision, recall, fscore, mcc, val_acc = evaluate(Y_test, predictions)

            val_loss = criterion(test_scores, Y_test).item()
            all_val_losses.append(val_loss)
            all_val_accs.append(val_acc)

            if val_acc > best_acc and val_acc < epoch_acc + 0.01:
                torch.save(model.state_dict(), 'cnn.pkl')
                print("Higher accuracy, New best model saved.")
                # print(pr.gamma)
                best_acc = val_acc
                best_pred_prob = pred_prob
                best_loss = val_loss
                torch.save(model.state_dict(), 'cnn_pr.pkl')
                torch.save(pr.state_dict(), 'pr.pkl')
                print("Prior models saved")
                

        if (epoch + 1) % 1 == 0:
            print('Epoch %d Time %s' % (epoch, get_time_string(elapsed_time)))
            print('T_loss %.3f\tT_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                epoch_loss, epoch_acc, epoch_pre, epoch_rec, epoch_fscore))
            print('V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                val_loss, val_acc, precision, recall, fscore))
    fpr_cnn, tpr_cnn, _ = roc_curve(Y_test.cpu(), pred_prob.cpu())
    AUC = auc(fpr_cnn, tpr_cnn)
    return [epoch_acc, epoch_pre, epoch_rec, epoch_fscore, AUC], [val_acc, precision, recall, fscore, AUC], best_acc

