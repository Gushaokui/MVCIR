import torch
from PIL import Image

from Network import Network
from metric import valid
import argparse
from dataloader import load_data
import scipy.io
# MNIST-USPS
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'UCI'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=50)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--K", default=5)
import warnings
warnings.filterwarnings("ignore")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims, args.feature_dim, class_num, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '3-mae' +'.pth')
model.load_state_dict(checkpoint)

print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
#result = torch.zeros((280, 280)).to(device)
for batch_idx, (xs, y, _) in enumerate(data_loader):
    for v in range(view):
        xs[v] = xs[v].to(device)
    ms = model.forward_mask(xs)
    qs, xr, zs = model(xs)
    qs, xrs, z = model(ms)
    X1 = ms[0].cpu().detach().numpy()
    X2 = ms[1].cpu().detach().numpy()
    X3 = ms[2].cpu().detach().numpy()
    Y = y.numpy()
    # data = {'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y}
    # scipy.io.savemat('Fashion1.mat', data)
valid(model, device, dataset, view, data_size, class_num, eval_h=True)
