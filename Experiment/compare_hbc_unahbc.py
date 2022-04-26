import os, sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, TensorDataset
from torchsummary import summary
import copy
import torchvision
############################
from setting import args_parser
from Dataset import datasets, constants
from Task import task
from Trainer import trainers
from Model import models
from baseline import get_expname
import Logger


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def compare_fairness(args, model, param_G, test_dataset, betas=0):
    # betas代表了是谁是固定值，谁是变量
    if betas == 0:
        compare_path = "./" + constants.DATASET + "_hbc_unahbc_compare_ks_ky.csv"
    else:
        compare_path = "./" + constants.DATASET + "_hbc_unahbc_compare_beta_s.csv"
    save_dir = args.root_dir + "/results/hbc/" + constants.DATASET + "/" + get_expname("hbc") + "/"
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    save_dir = args.root_dir + "/results/hbc/" + constants.DATASET + "/" + get_expname() + "/"
    param_G.load_state_dict(
        torch.load(save_dir + "best_param_G.pt", map_location=torch.device(args.device)))

    test_data_loader = DataLoader(
        TensorDataset(torch.Tensor(test_dataset[0]), torch.Tensor(test_dataset[1]).long()),
        batch_size=len(test_dataset[0]) // 50, shuffle=False, drop_last=False)

    hbc_target_acc, hbc_sensitive_acc = task.evaluate_acc_par(args, model, param_G, test_data_loader)
    fair_features_count, fair_features_correct = task.evaluate_fairness_par(args, model, param_G, test_data_loader)
    fair_acc = task.prepare_fair_parameter(fair_features_count, fair_features_correct)
    hbc_efg = fair_acc[3]

    save_dir = args.root_dir + "/results/unawarenessHbc/" + constants.DATASET + "/" + get_expname() + "/"
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    unahbc_target_acc, unahbc_sensitive_acc = task.evaluate_acc_par(args, model, param_G, test_data_loader)
    fair_features_count, fair_features_correct = task.evaluate_fairness_par(args, model, param_G, test_data_loader)
    fair_acc = task.prepare_fair_parameter(fair_features_count, fair_features_correct)
    unahbc_efg = fair_acc[3]
    Logger.compare_hbc_unahbc_summary(compare_path, [hbc_target_acc.item(), unahbc_target_acc.item()],
                                      [hbc_sensitive_acc.item(), unahbc_sensitive_acc.item()], [hbc_efg, unahbc_efg])


def compare_entropy(args, model, param_G, test_dataset, betas=0):
    # betas代表了是谁是固定值，谁是变量
    if betas == 0:
        compare_path = "./" + constants.DATASET + "_hbc_unahbc_compare_ks_ky_entropy.csv"
    else:
        compare_path = "./" + constants.DATASET + "_hbc_unahbc_compare_beta_s_entropy.csv"
    save_dir = args.root_dir + "/results/hbc/" + constants.DATASET + "/" + get_expname("hbc") + "/"
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    save_dir = args.root_dir + "/results/hbc/" + constants.DATASET + "/" + get_expname() + "/"
    param_G.load_state_dict(
        torch.load(save_dir + "best_param_G.pt", map_location=torch.device(args.device)))

    test_data_loader = DataLoader(
        TensorDataset(torch.Tensor(test_dataset[0]), torch.Tensor(test_dataset[1]).long()),
        batch_size=len(test_dataset[0]) // 50, shuffle=False, drop_last=False)

    hbc_target_acc, hbc_sensitive_acc = task.evaluate_acc_par(args, model, param_G, test_data_loader)
    hbc_yhat_s_entropy = task.evaluate_loss_s_entropy(args, model,param_G, test_data_loader)

    save_dir = args.root_dir + "/results/unawarenessHbc/" + constants.DATASET + "/" + get_expname() + "/"
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    unahbc_target_acc, unahbc_sensitive_acc = task.evaluate_acc_par(args, model, param_G, test_data_loader)
    unahbc_yhat_s_entropy = task.evaluate_loss_s_entropy(args, model,param_G, test_data_loader)

    Logger.compare_unawareness_unawarenessHbc_entropy(compare_path, [hbc_target_acc.item(), unahbc_target_acc.item()],
                                                      [hbc_sensitive_acc.item(), unahbc_sensitive_acc.item()],
                                                      [hbc_yhat_s_entropy, unahbc_yhat_s_entropy])


def train_diff_ky_ks_model(args):
    train_dataset = ()
    test_dataset = ()
    torch.manual_seed(constants.RANDOM_SEED)
    for k_y in range(2, 6):
        constants.K_Y = k_y
        for k_s in range(2, 6):
            constants.K_S = k_s
            ## Fetch the datasets
            if constants.DATASET == "utk_face":
                (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(args)
                train_labels, test_labels = datasets.prepare_labels(train_labels, test_labels)
                train_labels = train_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
                test_labels = test_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
                if args.task == "train_unawareness":
                    shape = train_labels[:, 1].shape
                    train_labels[:, 1] = np.zeros(shape)
                train_dataset = (train_images, train_labels)
                test_dataset = (test_images, test_labels)
            elif constants.DATASET == "celeba":
                (train_images, train_labels), (valid_images, valid_labels), (
                    test_images, test_labels) = datasets.get_dataset(
                    args)
                train_labels, valid_labels, test_labels = datasets.prepare_labels(train_labels, test_labels,
                                                                                  valid_labels)
                train_dataset = ((train_images, train_labels), (valid_images, valid_labels))
                test_dataset = (test_images, test_labels)

            ## For logging
            exp_name = get_expname()

            log_dir = args.root_dir + constants.result_path + constants.DATASET + "/" + exp_name + "/"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            Logger.set_logger(log_dir + "summary.log")

            print("\n*** Dataset's Info")
            print("Training")
            task.print_dataset_info(train_dataset)
            print("Testing")
            task.print_dataset_info(test_dataset)

            ## Model

            model = models.Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)
            model.to(args.device)
            summary(model, input_size=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE), device=args.device)
            param_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
            param_G.to(args.device)
            summary(param_G, input_size=(constants.K_Y,), device=args.device)

            if args.task == "train" or args.task == "train_unawareness":
                pass
            elif args.task == "compare":
                compare_entropy(args, model, param_G, test_dataset, 0)


def train_diff_beta_s(args):
    train_dataset = ()
    test_dataset = ()
    torch.manual_seed(constants.RANDOM_SEED)
    for beta_s in range(1, 6):
        constants.BETA_S = beta_s
        constants.BETA_Y = 10 - constants.BETA_S
        if constants.DATASET == "utk_face":
            (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(args)
            train_labels, test_labels = datasets.prepare_labels(train_labels, test_labels)
            train_labels = train_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
            test_labels = test_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
            if args.task == "train_unawareness":
                shape = train_labels[:, 1].shape
                train_labels[:, 1] = np.zeros(shape)
            train_dataset = (train_images, train_labels)
            test_dataset = (test_images, test_labels)
        elif constants.DATASET == "celeba":
            (train_images, train_labels), (valid_images, valid_labels), (
                test_images, test_labels) = datasets.get_dataset(
                args)
            train_labels, valid_labels, test_labels = datasets.prepare_labels(train_labels, test_labels,
                                                                              valid_labels)
            train_dataset = ((train_images, train_labels), (valid_images, valid_labels))
            test_dataset = (test_images, test_labels)

        ## For logging
        exp_name = get_expname()

        log_dir = args.root_dir + constants.result_path + constants.DATASET + "/" + exp_name + "/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        Logger.set_logger(log_dir + "summary.log")

        print("\n*** Dataset's Info")
        print("Training")
        task.print_dataset_info(train_dataset)
        print("Testing")
        task.print_dataset_info(test_dataset)

        ## Model

        model = models.Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)
        model.to(args.device)
        summary(model, input_size=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE), device=args.device)
        param_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
        param_G.to(args.device)
        summary(param_G, input_size=(constants.K_Y,), device=args.device)
        if args.task == "train":
            pass
        elif args.task == "compare":
            compare_entropy(args, model, param_G, test_dataset, 1)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = args_parser()  ## Reading the input arguments (see setting.py)
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    args.root_dir = ".."
    args.run_name = "baseline"
    args.task = "train"
    global constants

    constants.RANDOM_SEED = 0

    constants.BETA_Y = 5

    # 比较得到betas=3时，不同kyks下hbc与unahbc模型的区别
    constants.BETA_S = 3
    constants.BETA_Y = 7
    args.task = "compare"
    train_diff_ky_ks_model(args)

    # 比较固定KyKs后,不同beta_s下hbc与unahbc模型的区别
    for ky, ks in [(4, 3), (5, 2)]:
        constants.K_Y = ky
        constants.K_S = ks
        train_diff_beta_s(args)












