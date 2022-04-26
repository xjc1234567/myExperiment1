import torch
import collections
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Dataset import constants
from Task import task


def  get_data_loader(args, dataset, train=True):
    if constants.DATASET == "utk_face":
        if train:
            _x = dataset[0][constants.VALID_SHARE:]
            _y = dataset[1][constants.VALID_SHARE:]
            _x, _y = torch.Tensor(_x), torch.Tensor(_y).long()
            _xy = TensorDataset(_x, _y)
            data_loader = DataLoader(_xy, batch_size=args.server_batch,
                                     shuffle=True, drop_last=True)
        else:
            _x = dataset[0][:constants.VALID_SHARE]
            _y = dataset[1][:constants.VALID_SHARE]
            _x, _y = torch.Tensor(_x), torch.Tensor(_y).long()
            _xy = TensorDataset(_x, _y)
            data_loader = DataLoader(_xy, batch_size=args.server_batch,
                                     shuffle=False, drop_last=False)
        return data_loader
    elif constants.DATASET == "celeba":
        if train:
            _xy = TensorDataset(torch.Tensor(dataset[0][0]), torch.Tensor(dataset[0][1]).long())
            data_loader = DataLoader(_xy, batch_size=args.server_batch,
                                     shuffle=True, drop_last=True)
        else:
            _xy = TensorDataset(torch.Tensor(dataset[1][0]), torch.Tensor(dataset[1][1]).long())
            data_loader = DataLoader(_xy, batch_size=args.server_batch,
                                     shuffle=False, drop_last=False)
        return data_loader


def train_model_par(args, model_F, model_G, dataset, save_path=None):
    optimizer_F = optim.Adam(model_F.parameters(), lr=args.server_lr)
    optimizer_G = optim.Adam(model_G.parameters(), lr=args.server_lr)

    nll_loss_fn = nn.NLLLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    train_epoch_loss_Y, train_epoch_acc_Y, valid_epoch_acc_Y = [], [], []
    train_epoch_loss_S, train_epoch_acc_S, valid_epoch_acc_S = [], [], []
    train_epoch_loss_E = []
    best_valid_acc_Y = 0.
    for epoch in range(args.server_epochs):
        ## Training
        trainloader = get_data_loader(args, dataset, train=True)
        train_batch_loss_Y, train_batch_acc_Y = [], []
        train_batch_loss_S, train_batch_acc_S = [], []
        train_batch_loss_E = []
        model_F.train()
        for batch_id, (images, labels) in enumerate(trainloader):
            images, labels = images.to(args.device), labels.to(args.device)
            test_label = torch.zeros(100,dtype=torch.long)
            test_label = test_label.to(args.device)
            #### Training model_G
            if constants.BETA_S != 0:
                model_F.eval()
                model_G.train()
                optimizer_G.zero_grad()
                out_y = model_F(images)
                out_s = model_G(out_y)
                loss_s = ce_loss_fn(out_s, labels[:, 1])
                loss_s.backward()
                optimizer_G.step()
                model_G.eval()
                model_F.train()
            #### Training model_F
            optimizer_F.zero_grad()
            out_y = model_F(images)
            if constants.SOFTMAX:
                out_y = out_y + 1e-16
                loss_F = nll_loss_fn(torch.log(out_y), labels[:, 0])
            else:
                loss_F = ce_loss_fn(out_y, labels[:, 0])

            if constants.BETA_X != 0:
                if constants.SOFTMAX:
                    sftmx = out_y
                else:
                    sftmx = nn.Softmax(dim=1)(out_y)
                    sftmx = sftmx + 1e-16
                loss_E = -torch.mean(torch.sum(sftmx * torch.log(sftmx), dim=1))
                train_batch_loss_E.append(loss_E.item())

            if constants.BETA_S != 0:
                out_s = model_G(out_y)
                loss_G = ce_loss_fn(out_s, labels[:, 1])
                train_batch_loss_S.append(loss_G.item())
                train_batch_acc_S.append(torch.mean((out_s.max(1)[1] == labels[:, 1]).float()))

            if constants.BETA_X != 0 and constants.BETA_S != 0:
                loss = constants.BETA_Y * loss_F + \
                       constants.BETA_S * loss_G + \
                       constants.BETA_X * loss_E
            elif constants.BETA_S != 0:
                loss = constants.BETA_Y * loss_F + \
                       constants.BETA_S * loss_G
            elif constants.BETA_X != 0:
                loss = constants.BETA_Y * loss_F + \
                       constants.BETA_X * loss_E
            else:
                loss = constants.BETA_Y * loss_F

            loss.backward()
            optimizer_F.step()

            ####
            train_batch_loss_Y.append(loss_F.item())
            train_batch_acc_Y.append(torch.mean((out_y.max(1)[1] == labels[:, 0]).float()))

        train_epoch_loss_Y.append(sum(train_batch_loss_Y) / len(train_batch_loss_Y))
        train_epoch_acc_Y.append(sum(train_batch_acc_Y) / len(train_batch_acc_Y) * 100)
        if train_batch_loss_S:
            train_epoch_loss_S.append(sum(train_batch_loss_S) / len(train_batch_loss_S))
            train_epoch_acc_S.append(sum(train_batch_acc_S) / len(train_batch_acc_S) * 100)
        if train_batch_loss_E:
            train_epoch_loss_E.append(sum(train_batch_loss_E) / len(train_batch_loss_E))

        ## Validation
        validloader = get_data_loader(args, dataset, train=False)
        acc_Y, acc_S = task.evaluate_acc_par(args, model_F, model_G, validloader)
        valid_epoch_acc_Y.append(acc_Y)
        valid_epoch_acc_S.append(acc_S)

        constants.log.info("_________ Epoch: %d", epoch + 1)
        if save_path:
            wy = constants.BETA_Y / (constants.BETA_Y + constants.BETA_S)
            ws = constants.BETA_S / (constants.BETA_Y + constants.BETA_S)
            current_w_vacc = wy * valid_epoch_acc_Y[-1] + ws * valid_epoch_acc_S[-1]
            if current_w_vacc > best_valid_acc_Y:
                best_valid_acc_Y = current_w_vacc
                torch.save(model_F.state_dict(), save_path + "best_model.pt")
                torch.save(model_G.state_dict(), save_path + "best_param_G.pt")
                constants.log.info("**** Best Acc Y on Epoch {} is {:.2f}".format(epoch + 1, best_valid_acc_Y))
        constants.log.info("Train Loss Y: {:.5f}, \nTrain Acc Y: {:.2f}".format(train_epoch_loss_Y[-1],
                                                                                train_epoch_acc_Y[-1]))
        constants.log.info("Valid Acc Y: {:.2f}".format(valid_epoch_acc_Y[-1]))
        if train_epoch_loss_S:
            constants.log.info("Train Loss S: {:.5f}, \nTrain Acc S: {:.2f}".format(train_epoch_loss_S[-1],
                                                                                    train_epoch_acc_S[-1]))
            constants.log.info("Valid Acc S: {:.2f}".format(valid_epoch_acc_S[-1]))
        if train_epoch_loss_E:
            constants.log.info("Train Loss Entropy: {:.5f}".format(train_epoch_loss_E[-1]))

    return model_F, model_G


def train_model_decode_G(args,model_F, decode_G, dataset, save_path=None):
    model_F.eval()
    optimizer_G = optim.Adam(decode_G.parameters(), lr=args.server_lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    train_epoch_loss_S, train_epoch_acc_S, valid_epoch_acc_S = [], [], []
    best_valid_acc_Y = 0.
    for epoch in range(args.server_epochs):
        ## Training
        trainloader = get_data_loader(args, dataset, train=True)
        train_batch_loss_S, train_batch_acc_S = [], []
        train_batch_loss_E = []
        for batch_id, (images, labels) in enumerate(trainloader):
            images, labels = images.to(args.device), labels.to(args.device)
            yhat = model_F(images)

            #### Training model_G
            decode_G.train()
            optimizer_G.zero_grad()
            out_s = decode_G(yhat)
            loss_s = ce_loss_fn(out_s, labels[:, 1])
            loss_s.backward()
            optimizer_G.step()
            decode_G.eval()
            loss_G= ce_loss_fn(out_s, labels[:, 1])
            train_batch_loss_S.append(loss_G.item())
            train_batch_acc_S.append(torch.mean((out_s.max(1)[1] == labels[:, 1]).float()))

            train_epoch_loss_S.append(sum(train_batch_loss_S) / len(train_batch_loss_S))
            train_epoch_acc_S.append(sum(train_batch_acc_S) / len(train_batch_acc_S) * 100)

        ## Validation
        validloader = get_data_loader(args, dataset, train=False)
        acc_Y, acc_S = task.evaluate_acc_par(args, model_F,decode_G, validloader)
        valid_epoch_acc_S.append(acc_S)

        constants.log.info("_________ Epoch: %d", epoch + 1)
        if save_path:
            current_w_vacc = valid_epoch_acc_S[-1]
            if current_w_vacc > best_valid_acc_Y:
                best_valid_acc_Y = current_w_vacc
                torch.save(decode_G.state_dict(), save_path + "best_decode_param_G.pt")
                constants.log.info("**** Best Acc Y on Epoch {} is {:.2f}".format(epoch + 1, best_valid_acc_Y))
        if train_epoch_loss_S:
            constants.log.info("Train Loss S: {:.5f}, \nTrain Acc S: {:.2f}".format(train_epoch_loss_S[-1],
                                                                                train_epoch_acc_S[-1]))
            constants.log.info("Valid Acc S: {:.2f}".format(valid_epoch_acc_S[-1]))

    return decode_G

