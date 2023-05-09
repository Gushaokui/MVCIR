import torch.nn as nn
from torch.nn.functional import normalize
import torch
from scipy.special import comb
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, feature_dim)
        )
        nn.init.xavier_normal_(self.encoder[4].weight)
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

def xavier_init_(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class SoftmaxLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Softmax(dim=1)
        )
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(input_size[views], input_size[views]) for views in range(view)])
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.contrastive_module = nn.ModuleList(
            [SoftmaxLayer(feature_dim, class_num) for views in range(view)])
        self.param = nn.Parameter(torch.ones(view))
        self.view = view

    def forward_mask(self, xs):
        FeatureInfo,ms= dict(),dict()
        for v in range(self.view):
            x = xs[v]
            FeatureInfo[v] = torch.sigmoid(self.FeatureInforEncoder[v](x))
            ms[v] = FeatureInfo[v] * xs[v]
        return ms

    def forward(self, xs):
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.contrastive_module[v](z)
            xr = self.decoders[v](z)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return qs, xrs, zs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        qst = None
        i = 1
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            if i==1:
                qst = z
                i = i+1
            else:
                qst = torch.cat((qst, z), 1)
            q = self.contrastive_module[v](z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds, qst

    def get_parpam(self):

        param = torch.softmax(self.param, 0)

        return param