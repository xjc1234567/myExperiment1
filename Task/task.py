import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import torch
from torch import nn
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import torchvision
from Dataset import constants
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import collections

def get_modelF_output_with_slabel(args,model,dataloader):
    model.eval()
    yhats = torch.Tensor()
    slabels = torch.Tensor()
    for batch_id, (images, labels) in enumerate(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)
        yhat = model(images)
        yhats = torch.cat((yhats,yhat),dim=0)
        slabels = torch.cat((slabels,labels[:,1]),dim=0)
    return (yhats,slabels)



def evaluate_acc_par(args, model, param_G, dataloader):
    model.eval()
    param_G.eval()
    valid_batch_acc_1, valid_batch_acc_2 = [], []
    y_true_1, y_pred_1, y_prob_1 = [], [], []
    y_true_2, y_pred_2, y_prob_2 = [], [], []
    n_samples = 0
    for batch_id, (images, labels) in enumerate(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)
        output_valid_1 = model(images)
        output_valid_2 = param_G(output_valid_1)
        predictions_1 = output_valid_1.max(1)[1]
        predictions_2 = output_valid_2.max(1)[1]
        current_acc_1 = torch.sum((predictions_1 == labels[:, 0]).float())
        current_acc_2 = torch.sum((predictions_2 == labels[:, 1]).float())
        valid_batch_acc_1.append(current_acc_1)
        valid_batch_acc_2.append(current_acc_2)
        n_samples += len(labels)
        y_true_1 = y_true_1 + labels[:, 0].tolist()
        y_pred_1 = y_pred_1 + predictions_1.tolist()
        if constants.SOFTMAX:
            y_prob_1 = y_prob_1 + ((output_valid_1).detach().cpu().numpy()).tolist()
        else:
            y_prob_1 = y_prob_1 + (nn.Softmax(dim=1)(output_valid_1).detach().cpu().numpy()).tolist()
        y_true_2 = y_true_2 + labels[:, 1].tolist()
        y_pred_2 = y_pred_2 + predictions_2.tolist()
        y_prob_2 = y_prob_2 + (nn.Softmax(dim=1)(output_valid_2).detach().cpu().numpy()).tolist()
    acc_1 = (sum(valid_batch_acc_1) / n_samples) * 100
    acc_2 = (sum(valid_batch_acc_2) / n_samples) * 100

    return acc_1, acc_2

def evaluate_loss_s_entropy(args, model,param_G, dataloader):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    loss = 0
    for batch_id, (images, labels) in enumerate(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)
        yhat = model(images)
        shat = param_G(yhat)
        loss_s = ce_loss_fn(shat, labels[:, 1])
        loss = loss + loss_s.item()
    return loss

def evaluate_loss_s_decode(args, param_G, dataloader):
    param_G.eval()
    valid_batch_acc_s = []
    n_samples = 0
    for batch_id, (yhats, labels) in enumerate(dataloader):
        yhats, labels = yhats.to(args.device), labels.to(args.device)
        output_valid_s = param_G(yhats)
        predictions_s = output_valid_s.max(1)[1]
        current_acc_s = torch.sum((predictions_s == labels[:, 1]).float())
        valid_batch_acc_s.append(current_acc_s)
        n_samples += len(labels)
    acc_s = (sum(valid_batch_acc_s) / n_samples) * 100
    return acc_s

def evaluate_fairness_par(args, model, param_G, dataloader):
    model.eval()
    param_G.eval()
    target_correct = 0
    fair_features_count = collections.Counter()
    fair_features_correct = collections.Counter()
    group_count = collections.Counter()
    group_correct = collections.Counter()
    likelihood_count = collections.Counter()
    n_samples = 0
    for batch_id, (images, labels) in enumerate(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)
        output_valid_1 = model(images)
        output_valid_2 = param_G(output_valid_1)
        predictions_1 = output_valid_1.max(1)[1]
        predictions_2 = output_valid_2.max(1)[1]
        target = labels[:, 0]
        sensitive = labels[:, 1]
        target_correct += predictions_1.eq(target).cpu().sum().item()
        _update_counters(predictions_1, target, sensitive, fair_features_count, fair_features_correct,
                         group_count, group_correct, likelihood_count)

        n_samples += len(labels)

    return fair_features_count, fair_features_correct

def prepare_fair_parameter(fair_features_count, fair_features_correct):
    fair_acc_lst = []
    min_fair_label_count = 10000000
    for fair_label, counter in fair_features_correct.items():
        fair_label_count = fair_features_count[fair_label]
        min_fair_label_count = fair_label_count if fair_label_count < min_fair_label_count else min_fair_label_count
        fair_acc = 1. * counter / fair_label_count
        fair_acc_lst.append(fair_acc)
        constants.log.info(f'Fair-label: {fair_label}'
                           f'\tAccuracy: {counter}/{fair_label_count} ({fair_acc:.4f})')
    fair_acc_lst.sort()
    EFG = fair_acc_lst[-1] - fair_acc_lst[0]
    constants.log.info(f'Fairness Epsilon: {constants.fair_epsilon:.4f}')
    constants.log.info(f'Fairness delta: {constants.fair_delta:.4f}')
    constants.log.info(f'Empirical Fairness Gap: {EFG:.4f}')
    constants.log.info(f'Minimum Sensitive Number: {min_fair_label_count}')
    passed = False
    threshold = fairness_threshold(EFG, constants.K_Y, constants.K_S, constants.fair_epsilon,
                                   constants.fair_delta)
    constants.log.info(f'Fairness Threshold: {threshold:.4f}')
    if EFG <= constants.fair_epsilon:
        passed = min_fair_label_count > threshold
    constants.log.info(f'Pass Fairness Test: {passed}')
    return [constants.fair_epsilon,constants.fair_delta,min_fair_label_count,EFG,threshold,passed]


def _update_counters(pred, target, fair_features, fair_features_count, fair_features_correct, group_count=None,
                     group_correct=None, likelihood_count=None):
    for fair_label in fair_features.unique():  # add fair group acc and count:
        fair_label = fair_label.item()
        mask = fair_features == fair_label
        label_pred = pred[mask]
        label_target = target[mask]
        fair_features_count[fair_label] += label_pred.shape[0]
        fair_features_correct[fair_label] += label_pred.eq(label_target.view_as(label_pred)).cpu().sum().item()
    for fair_label in fair_features.unique():
        fair_label = fair_label.item()
        if group_count is None:
            continue
        for label in target.unique():  # add group (fair + label) acc and count:
            label = label.item()
            mask = (fair_features == fair_label) & (target == label)
            label_pred = pred[mask]
            label_target = target[mask]
            group_count[(fair_label, label)] += label_pred.shape[0]
            group_correct[(fair_label, label)] += label_pred.eq(label_target.view_as(label_pred)).cpu().sum().item()
        if likelihood_count is None:
            continue
        pred = pred.squeeze()
        for pred_label in pred.unique():  # add (fair) likelihood count:
            pred_label = pred_label.item()
            mask = (fair_features == fair_label) & (pred == pred_label)
            label_pred = pred[mask]
            likelihood_count[(fair_label, pred_label)] += label_pred.shape[0]


def evaluate_acc_reg(args, model, dataloader, cf_mat=False, roc=False, preds=False, beTau=constants.REGTAU):
    model.eval()
    valid_batch_acc_1, valid_batch_acc_2 = [], []
    y_true_1, y_pred_1, y_prob_1 = [], [], []
    y_true_2, y_pred_2, y_prob_2 = [], [], []
    n_samples = 0
    for batch_id, (images, labels) in enumerate(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)
        output_valid_1 = model(images)
        predictions_1 = output_valid_1.max(1)[1]

        if constants.SOFTMAX:
            output_valid_2 = output_valid_1
        else:
            output_valid_2 = nn.Softmax(dim=1)(output_valid_1)
        output_valid_2 = output_valid_2 + 1e-16
        entropies = -(torch.sum(output_valid_2 * torch.log(output_valid_2), dim=1))
        predictions_2 = torch.where(entropies >= beTau, torch.tensor(1.).to(args.device),
                                    torch.tensor(0.).to(args.device))

        current_acc_1 = torch.sum((predictions_1 == labels[:, 0]).float())
        current_acc_2 = torch.sum((predictions_2 == labels[:, 1]).float())
        valid_batch_acc_1.append(current_acc_1)
        valid_batch_acc_2.append(current_acc_2)
        n_samples += len(labels)
        y_true_1 = y_true_1 + labels[:, 0].tolist()
        y_pred_1 = y_pred_1 + predictions_1.tolist()
        y_prob_1 = y_prob_1 + output_valid_2.detach().cpu().numpy().tolist()
        y_true_2 = y_true_2 + labels[:, 1].tolist()
        y_pred_2 = y_pred_2 + predictions_2.tolist()
        entropies = entropies.detach().cpu().numpy()
        y_prob_2 = y_prob_2 + np.concatenate((1 - entropies[:, np.newaxis], entropies[:, np.newaxis]), axis=1).tolist()
    acc_1 = (sum(valid_batch_acc_1) / n_samples) * 100
    acc_2 = (sum(valid_batch_acc_2) / n_samples) * 100
    y_true_2 = [int(x) for x in y_true_2]
    if roc:
        plt.figure(figsize=(5, 5))
        # skplt.metrics.
        plot_roc(y_true_1, y_prob_1,
                 plot_micro=False, plot_macro=False,
                 title=None,
                 cmap='prism',
                 figsize=(5, 5),
                 text_fontsize=14,
                 title_fontsize="large",
                 line_color=['r', 'b'],
                 line_labels=["male", "female"],
                 line_style=["-", "--"])
        plt.show()
        plt.figure(figsize=(5, 5))
        # skplt.metrics.
        plot_roc(y_true_2, y_prob_2,
                 plot_micro=False, plot_macro=False,
                 title=None,
                 cmap='prism',
                 figsize=(5, 5),
                 text_fontsize=14,
                 title_fontsize="large",
                 line_color=['m', 'g'],
                 line_labels=["white", "non-white"],
                 line_style=["-.", ":"])
        plt.show()
    if cf_mat:
        cf_1 = confusion_matrix(y_true_1, y_pred_1, normalize='true')
        cf_2 = confusion_matrix(y_true_2, y_pred_2, normalize='true')
        return acc_1, acc_2, cf_1, cf_2
    return acc_1, acc_2


def evaluate_acc_get_preds(args, model, param_G, dataloader):
    model.eval()
    param_G.eval()
    preds = []
    n_samples = 0
    for batch_id, (images, labels) in enumerate(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)
        preds = preds + ((model(images)).detach().cpu().numpy()).tolist()
    return preds


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


def print_dataset_info(dataset):
    print("Data Dimensions: ", dataset[0].shape)
    labels = np.array(dataset[1]).astype(int)
    _unique, _counts = np.unique(labels[:, 0], return_counts=True)
    print("Honest:\n", np.asarray((_unique, _counts)).T)
    _unique, _counts = np.unique(labels[:, 1], return_counts=True)
    print("Curious:\n", np.asarray((_unique, _counts)).T)


def imshow(imgs):
    img = torchvision.utils.make_grid(imgs)
    plt.figure(figsize=((len(imgs) + 1) * 5, 5))
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def fairness_threshold(efg,num_y,num_s,epsilon,delta):
    part1 = 2/(epsilon - efg)**2
    part2 = math.log(num_y*num_s*2/delta)
    threshold = part1*part2
    return threshold

