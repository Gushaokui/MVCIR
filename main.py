# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import argparse
import warnings
import random
import numpy as np
import os
from dataloader import load_data
from Network import Network
from loss import ClusterLoss
from torch.optim.lr_scheduler import StepLR
from metric import valid
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
import math
from scipy.special import comb
warnings.filterwarnings("ignore")
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

Dataname = 'UCI'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=100)
parser.add_argument("--con_epochs", default=0)
parser.add_argument("--tune_epochs", default=200)
parser.add_argument("--feature_dim", default=50)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--K", default=5)
parser.add_argument("--seed", default=10)
parser.add_argument("--alpha", default=1.7)
parser.add_argument("--lambda_", default=100)
parser.add_argument("--gamma", default=0.1)
parser.add_argument("--eta", default=1)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.

if args.dataset == "MNIST-USPS":
    args.tune_epochs = 50
    args.seed = 1
if args.dataset == "Fashion":
    args.K = 10
    args.batch_size = 512
    args.seed = 1
if args.dataset == "Caltech-2V":
    args.seed = 1
if args.dataset == "Caltech-3V":
    args.batch_size = 256
    args.lambda_ = 10
    args.gamma = 1
    args.seed = 1
if args.dataset == "Caltech-4V":
    args.seed = 1
if args.dataset == "Caltech-5V":
    args.seed = 1
if args.dataset == "UCI":
    args.tune_epochs = 100
    args.K = 7
    args.lambda_ = 10

# args.tune_epochs = 1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

args.K = args.K + 1

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch-1 {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    # zero = torch.zeros(args.batch_size).to(device)
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward(qs[v], qs[w]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch-2-0 {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

def JSdiv_train(epoch, class_num):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    KLdiv = torch.nn.KLDivLoss()
    alpha = args.alpha
    lambda_ = args.lambda_
    gamma = args.gamma
    eta = args.eta
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        q, _, zs = model(xs)
        forward_knn = graph(zs, epoch)
        for_matrix = complement_confidence(forward_knn, q)
        optimizer.zero_grad()
        ms = model.forward_mask(xs)
        qs, xrs, zs = model(ms)
        param = model.get_parpam()
        loss_list = []
        for v in range(view):
            loss_list.append(lambda_ * KLdiv(qs[v].log(), for_matrix[v]))
            if v < view - 1:
                M = torch.softmax((qs[v] * qs[v+1]) * alpha, dim=1)
                kl_p = F.kl_div(qs[v].log(), M)
                kl_q = F.kl_div(qs[v+1].log(), M)
                loss_list.append(gamma * 0.5 * (kl_q + kl_p))
            loss_list.append(eta * class_num * param[v] * mes(ms[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    # print(tot / (data_size // args.batch_size))
    # print('Epoch-2-2 {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
def graph(zs, epoch):
    d = torch.nn.PairwiseDistance()
    N = args.batch_size
    mask_1 = torch.ones(N, args.K)
    mask_0 = torch.zeros(N, N - args.K)
    mask = torch.cat((mask_1, mask_0), 1)
    mask = mask.bool()
    forward_knn = dict()
    dis = torch.zeros(N, N).to(device)
    for v in range(view):
        forward_knn[v] = d(zs[v].unsqueeze(1), zs[v].unsqueeze(0))
        dis = dis + forward_knn[v]**2
        tmin = torch.min(forward_knn[v], dim=1).values
        tmax = torch.max(forward_knn[v], dim=1).values
        tsub = tmax - tmin
        forward_knn[v].sub_(tmin[:, None]).div_(tsub[:, None])
        forward_knn[v] = forward_knn[v] + forward_knn[v].T - forward_knn[v] * forward_knn[v].T
        forward_knn[v] = forward_knn[v].argsort()
        forward_knn[v] = forward_knn[v][mask].reshape(N, -1)
    s = dis.argsort()[:, 1]
    for_knn = dict()
    knn = forward_knn[0].cpu().detach().numpy()
    for v in range(view-1):
        for_knn[v] = knn
        for_knn[v+1] = forward_knn[v+1].cpu().detach().numpy()
        knn = args.batch_size * torch.ones(N, args.K)
        knn = knn.numpy()
        for i in range(N):
            knn[i][0: len(np.intersect1d(for_knn[v][i], for_knn[v + 1][i]))] = np.intersect1d(for_knn[v][i], for_knn[v + 1][i])
    for i in range(N):
        if knn[:, 1][i] == N:
            knn[:, 1][i] = s[i]
    knn = torch.from_numpy(knn).long()
    return knn

def complement_confidence(forward_knn, qs):

    for_matrix= dict()
    for v in range(view):
        q = torch.cat((qs[v], torch.ones(1, class_num).to(device)), 0)
        for_matrix[v] = torch.softmax((q[forward_knn].cumprod(1))[:, args.K - 1], 1)
    return for_matrix

if not os.path.exists('./models'):
    os.makedirs('./models')

T = 1
for i in range(T):

    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    print("ROUND:{}".format(i+1))
    model = Network(view, dims, args.feature_dim, class_num, device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.NAdam(model.parameters(), lr=args.learning_rate)
    criterion = ClusterLoss(class_num, args.temperature_f, device).to(device)
    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.tune_epochs:
        contrastive_train(epoch)
        JSdiv_train(epoch, class_num)
        if epoch % 50 == 0:
            model.eval()
            print(epoch)
            with torch.no_grad():
                acc, nmi, pur, acc2, nmi2, pur2 = valid(model, device, dataset, view, data_size, class_num, eval_h=True)
            model.train()
        if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '3-mae' + '.pth')
            print('Saving..')
            acc, nmi, pur, acc2, nmi2, pur2= valid(model, device, dataset, view, data_size, class_num, eval_h=True)

        epoch += 1