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


def train_model(args,model,param_G,exp_name,train_dataset,test_dataset):
    save_dir = args.root_dir + constants.result_path + constants.DATASET + "/" + exp_name + "/"
    if not os.path.exists(save_dir):
        os.makedir(save_dir)
    model, param_G = trainers.train_model_par(args, model, param_G, train_dataset, save_dir)

    ## Test
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    param_G.load_state_dict(
        torch.load(save_dir + "best_param_G.pt", map_location=torch.device(args.device)))
    test_data_loader = DataLoader(
        TensorDataset(torch.Tensor(test_dataset[0]), torch.Tensor(test_dataset[1]).long()),
        batch_size=len(test_dataset[0]) // 50, shuffle=False, drop_last=False)
    eval_acc_1, eval_acc_2 = task.evaluate_acc_par(args, model, param_G,test_data_loader)
    constants.log.info("\n$$$ Test Accuracy of the BEST model 1 {:.2f}".format(eval_acc_1))
    constants.log.info("\n$$$ Test Accuracy of the BEST model 2 {:.2f}".format(eval_acc_2))

def train_decode_model(args,model_F,param_G,exp_name,train_dataset,test_dataset):
    save_dir = args.root_dir + constants.result_path + constants.DATASET + "/" + exp_name + "/"
    if not os.path.exists(save_dir):
        os.makedir(save_dir)
    model_F.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    param_G = trainers.train_model_decode_G(args, model_F, param_G, train_dataset, save_dir)
    test_data_loader = DataLoader(
        TensorDataset(torch.Tensor(test_dataset[0]), torch.Tensor(test_dataset[1]).long()),
        batch_size=len(test_dataset[0]) // 50, shuffle=False, drop_last=False)
    param_G.load_state_dict(torch.load(save_dir + "best_decode_param_G.pt", map_location=torch.device(args.device)))
    eval_acc_1, eval_acc_2 = task.evaluate_acc_par(args, model_F, param_G,test_data_loader)
    constants.log.info("\n$$$ Test Accuracy of the BEST model 1 {:.2f}".format(eval_acc_1))
    constants.log.info("\n$$$ Test Accuracy of the BEST model 2 {:.2f}".format(eval_acc_2))


def get_expname(mode="hbc"):
    if mode == "unawareness":
        return str(constants.HONEST) + "_" + str(constants.CURIOUS) + "_" + str(constants.K_Y) + \
                       "_" + str(constants.K_S) + "_" + str(2) + "_" + str(0) + \
                       "_" + str(5) + "_" +str(constants.SOFTMAX) + "/" + str(constants.IMAGE_SIZE) +\
                       "_" + str(constants.RANDOM_SEED)
    else:
        return str(constants.HONEST) + "_" + str(constants.CURIOUS) + "_" + str(constants.K_Y) + \
                       "_" + str(constants.K_S) + "_" + str(int(constants.BETA_X)) + "_" + str(int(constants.BETA_S)) + \
                       "_" + str(int(constants.BETA_Y)) + "_" +str(constants.SOFTMAX) + "/" + str(constants.IMAGE_SIZE) +\
                       "_" + str(constants.RANDOM_SEED)

def compare_param_G(args,model,changed_model,decode_G,test_dataset,betas=0):
    # betas代表了是谁是固定值，谁是变量
    if betas == 0:
        compare_path = "./" + constants.DATASET + "_compare_ks_ky.csv"
    else:
        compare_path = "./" + constants.DATASET + "_compare_beta_s.csv"
       
    test_data_loader = DataLoader(
        TensorDataset(torch.Tensor(test_dataset[0]), torch.Tensor(test_dataset[1]).long()),
        batch_size=len(test_dataset[0]) // 50, shuffle=False, drop_last=False)

    #hbc
    save_dir = args.root_dir + "/results/hbc/" + constants.DATASET + "/" + get_expname("hbc") + "/"
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    save_dir = args.root_dir + "/results/changed_hbc/" + constants.DATASET + "/" + get_expname("hbc") + "/"
    changed_model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))

    decode_G.load_state_dict(
        torch.load(save_dir + "best_decode_param_G.pt", map_location=torch.device(args.device)))
    

    hbc_tacc, hbc_sacc = task.evaluate_acc_par(args, model, decode_G, test_data_loader)
    changed_hbc_tacc, changed_hbc_sacc = task.evaluate_acc_par(args, changed_model, decode_G, test_data_loader)

    #unawareness
    save_dir = args.root_dir + "/results/unawareness/" + constants.DATASET + "/" + get_expname("unawareness") + "/"
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    save_dir = args.root_dir + "/results/changed_unawareness/" + constants.DATASET + "/" + get_expname("unawareness") + "/"
    changed_model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    decode_G.load_state_dict(
        torch.load(save_dir + "best_decode_param_G.pt", map_location=torch.device(args.device)))
    una_tacc, una_sacc = task.evaluate_acc_par(args, model, decode_G, test_data_loader)
    changed_una_tacc, changed_una_sacc = task.evaluate_acc_par(args, model, decode_G, test_data_loader)



    columns = ['K_Y','K_S','B_S','UNA_YACC','UNA_SACC','CHANGED_UNA_SACC',
               'HBC_YACC','HBC_SACC','CHANGED_HBC_SACC',]
    rows = [constants.K_Y,constants.K_S,constants.BETA_S,una_tacc.item(),una_sacc.item(),changed_una_sacc.item(),
            hbc_tacc.item(),hbc_sacc.item(),changed_hbc_sacc.item(),]
    Logger.summary.compare_decode_paramG(compare_path,rows,columns)

def train_diff_ky_ks_model(args):
    train_dataset = ()
    test_dataset = ()
    torch.manual_seed(constants.RANDOM_SEED)
    for k_y in range(2, 4):
        constants.K_Y = k_y
        for k_s in range(2, 4):
            constants.K_S = k_s
            ## Fetch the datasets
            if constants.DATASET == "utk_face":
                (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(args)
                train_labels, test_labels = datasets.prepare_labels(train_labels, test_labels)
                train_labels = train_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
                test_labels = test_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)

                if constants.DATASET_CHANGED:
                    shape = train_labels[:,1].shape
                    train_labels[:,1] = np.zeros(shape)

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
            if args.task == "train_unawareness" or args.task == "train_decode_unawareness":
                exp_name = get_expname("unawareness")
            else:
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
            decode_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
            decode_G.to(args.device)
            summary(decode_G, input_size=(constants.K_Y,), device=args.device)

            if args.task == "train" or args.task == "train_unawareness":
                train_model(args,model,decode_G,exp_name,train_dataset,test_dataset)
            elif args.task == "train_decode" or args.task == "train_decode_unawareness":
                train_decode_model(args,model,decode_G,exp_name,train_dataset, test_dataset)
            elif args.task == "compare":
                changed_model = models.Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)
                changed_model.to(args.device)
                compare_param_G(args,model,changed_model,decode_G,test_dataset,0)

def train_diff_beta_s(args):
    train_dataset = ()
    test_dataset = ()
    torch.manual_seed(constants.RANDOM_SEED)
    for beta_s in range(1, 5):
        constants.BETA_S = beta_s
        constants.BETA_Y = 10 - constants.BETA_S
        if constants.DATASET == "utk_face":
            (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(args)
            train_labels, test_labels = datasets.prepare_labels(train_labels, test_labels)
            train_labels = train_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
            test_labels = test_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)

            if constants.DATASET_CHANGED:
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
        if args.task == "train_unawareness" or args.task == "train_decode_unawareness":
            exp_name = get_expname("unawareness")
        else:
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
            train_model(args, model, param_G, exp_name, train_dataset, test_dataset)
        elif args.task == "train_decode" or args.task == "train_decode_unawareness":
            train_decode_model(args, model, param_G, exp_name, train_dataset, test_dataset)
        elif args.task == "compare":
            decode_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
            decode_G.to(args.device)
            compare_param_G(args, model, param_G, decode_G, test_dataset, 1)


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

    constants.BETA_X = 2
    constants.BETA_S = 0
    constants.BETA_Y = 5


    for i in range(0,2):
        # 训练una分类器模型,一共4*4=16个
        constants.DATASET_CHANGED = i
        constants.BETA_S = 0
        constants.BETA_Y = 5
        if constants.DATASET_CHANGED:
            constants.result_path = "/results/changed_unawareness/"
        else:
            constants.result_path = "/results/unawareness/"
        args.task = "train_unawareness"
        train_diff_ky_ks_model(args)

        # 补充训练各种情况下的unawareness的decode_param_G 4*4=16
        constants.result_path = "/results/unawareness/"
        args.task = "train_decode_unawareness"
        train_diff_ky_ks_model(args)

        # 训练beta_s=3的hbc模型，4*4=16个模型
        constants.BETA_S = 3
        constants.BETA_Y = 7
        if constants.DATASET_CHANGED:
            constants.result_path = "/results/changed_hbc/"
        else:
            constants.result_path = "/results/hbc/"
        args.task = " train"
        train_diff_ky_ks_model(args)


        # 训练各种K_Y,K_S下的hbc分类器的decode_param_G 4*4=16
        constants.BETA_X = 2
        constants.BETA_S = 3
        constants.BETA_Y = 7
        args.task = "train_decode"
        if constants.DATASET_CHANGED:
            constants.result_path = "/results/changed_hbc/"
        else:
            constants.result_path = "/results/hbc/"
        train_diff_ky_ks_model(args)






    '''
    #训练得到确定K_Y,K_S情况下不同beta_s下的hbc分类器 4*4=16
    args.task = "train"
    for ky,ks in [(2,2),(2,5),(5,2),(5,5)]:
        constants.K_Y = ky
        constants.K_S = ks
        constants.result_path = "/results/hbc/"
        args.task = "train"
        train_diff_beta_s(args)
    
    
    

    #训练各种BETA_S下的hbc分类器的decode_param_G 4*4=16
    for ky, ks in [(2,2),(2,5),(5,2),(5,5)]:
        constants.K_Y = ky
        constants.K_S = ks
        constants.result_path = "/results/hbc/"
        args.task = "train_decode"
        train_diff_beta_s(args)
    '''

    #比较得到betas=3时，不同kyks下una与hbc模型的区别
    constants.BETA_X = 2
    constants.BETA_S = 3
    constants.BETA_Y = 7
    args.task = "compare"
    train_diff_ky_ks_model(args)
    
    '''
    # 比较固定KyKs后,不同beta_s下una与unahbc模型的区别
    for ky, ks in [(2,2),(2,5),(5,2),(5,5)]:
        constants.K_Y = ky
        constants.K_S = ks
        args.task = "compare"
        train_diff_beta_s(args)
    '''















