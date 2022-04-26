import os,sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, TensorDataset
from torchsummary import summary
import copy
import torchvision
############################
from setting import args_parser
from Dataset import datasets,constants
from Task import task
from Trainer import trainers
from Model import  models
import Logger
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = args_parser()  ## Reading the input arguments (see setting.py)
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    args.root_dir = ".."
    args.run_name = "test_beta_X"
    args.task = "fairness"
    global constants
    constants.BETA_Y = 5
    constants.BETA_S = 0
    constants.BETA_X = 0
    constants.summary_file = "beta_X_summary.csv"
    for beta_x in range(2,11,3):
        constants.BETA_X = beta_x
        for k_y in range(2, 5):
            constants.K_Y = k_y
            for k_s in range(2, 5):
                constants.K_S = k_s
                ## Fetch the datasets
                if constants.DATASET == "utk_face":
                    (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(args)
                    train_labels, test_labels = datasets.prepare_labels(train_labels, test_labels)
                    train_labels = train_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
                    test_labels = test_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
                    train_dataset = (train_images, train_labels)
                elif constants.DATASET == "celeba":
                    (train_images, train_labels), (valid_images, valid_labels), (
                    test_images, test_labels) = datasets.get_dataset(
                        args)
                    train_labels, valid_labels, test_labels = datasets.prepare_labels(train_labels, test_labels,
                                                                                      valid_labels)
                    train_dataset = ((train_images, train_labels), (valid_images, valid_labels))

                ## For logging
                exp_name = str(constants.HONEST) + "_" + str(constants.CURIOUS) + "_" + str(constants.K_Y) + \
                           "_" + str(constants.K_S) + "_" + str(int(constants.BETA_X)) + "_" + str(int(constants.BETA_S)) + \
                           "_" + str(int(constants.BETA_Y)) + "_" +str(constants.SOFTMAX) + "/" + str(constants.IMAGE_SIZE) +\
                           "_" + str(constants.RANDOM_SEED)

                log_dir = args.root_dir + constants.result_path + constants.DATASET + "/" + exp_name + "/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                Logger.set_logger(log_dir + "summary.log")

                print("\n*** Dataset's Info")
                print("Training")
                task.print_dataset_info((train_images, train_labels))
                print("Testing")
                task.print_dataset_info((test_images, test_labels))

                ## Model
                model = models.Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)
                model.to(args.device)
                fair_acc_lst = []
                summary(model, input_size=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE), device=args.device)

                if args.attack == "parameterized" and args.task == "train":
                    param_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
                    param_G.to(args.device)
                    summary(param_G, input_size=(constants.K_Y,), device=args.device)

                    save_dir = args.root_dir + constants.result_path + constants.DATASET + "/" + exp_name + "/"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    model, param_G = trainers.train_model_par(args, model, param_G, train_dataset, save_dir)

                    ## Test
                    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
                    param_G.load_state_dict(
                        torch.load(save_dir + "best_param_G.pt", map_location=torch.device(args.device)))
                    test_data_loader = DataLoader(
                        TensorDataset(torch.Tensor(test_images), torch.Tensor(test_labels).long()),
                        batch_size=len(test_images) // 50, shuffle=False, drop_last=False)
                    eval_acc_1, eval_acc_2, cf_mat_1, cf_mat_2 = task.evaluate_acc_par(args, model, param_G,
                                                                                       test_data_loader,
                                                                                       cf_mat=True, roc=False)
                    constants.log.info("\n$$$ Test Accuracy of the BEST model 1 {:.2f}".format(eval_acc_1))
                    print("     Confusion Matrix 1:\n", (cf_mat_1 * 100).round(2))
                    constants.log.info("\n$$$ Test Accuracy of the BEST model 2 {:.2f}".format(eval_acc_2))
                    print("     Confusion Matrix 2:\n", (cf_mat_2 * 100).round(2))

                    ### Optional: to report the avg entropy
                    yh = task.evaluate_acc_par(args, model, param_G, test_data_loader, preds=True)
                    yh = torch.tensor(yh)
                    yh_entropies = (-torch.sum(yh * torch.log2(yh), dim=1))
                    norm_ent = torch.linalg.norm(yh_entropies, ord=1) / len(yh_entropies)
                    constants.log.info("\n$$$ The average of the entropy of classifier’soutput {:.4f}".format(norm_ent))

                if args.task == "fairness":
                    param_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
                    param_G.to(args.device)
                    summary(param_G, input_size=(constants.K_Y,), device=args.device)
                    save_dir = args.root_dir + constants.result_path + constants.DATASET + "/" + exp_name + "/"
                    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
                    param_G.load_state_dict(
                        torch.load(save_dir + "best_param_G.pt", map_location=torch.device(args.device)))

                    '''
                    total_image = np.concatenate((train_images, test_images), axis=0)
                    total_labels = np.concatenate((train_labels, test_labels), axis=0)
                    total_data_loader = DataLoader(TensorDataset(torch.Tensor(total_image), torch.Tensor(total_labels).long()),
                                             batch_size=len(test_images) // 50, shuffle=False, drop_last=False)
                    '''
                    train_data_loader = DataLoader(TensorDataset(torch.Tensor(train_images), torch.Tensor(train_labels).long()),
                                             batch_size=len(test_images) // 50, shuffle=False, drop_last=False)
                    test_data_loader = DataLoader(
                        TensorDataset(torch.Tensor(train_images), torch.Tensor(train_labels).long()),
                        batch_size=len(test_images) // 50, shuffle=False, drop_last=False)

                    train_acc_1,train_acc_2 = task.evaluate_acc_par(args, model,param_G,train_data_loader)
                    test_acc_1, test_acc_2 = task.evaluate_acc_par(args, model, param_G, test_data_loader)
                    fair_features_count, fair_features_correct =task.evaluate_fairness_par(args, model, param_G,test_data_loader)
                    fair_acc = task.prepare_fair_parameter(fair_features_count, fair_features_correct)
                    summary_path = "."
                    Logger.write_experiment_summary(args,summary_path,[train_acc_1,train_acc_2],[test_acc_1,test_acc_2],fair_acc)








