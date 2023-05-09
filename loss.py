import torch
import torch.nn as nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device
        self.batch_size = 128
        self.temperature_f = 1.
        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def mask_correlated_instances(self, class_num):
        N = class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        mask = mask.bool()
        return mask

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    # def forward_instance(self,qsc,cof):
    #     N = self.batch_size
    #
    #     sim = torch.matmul(qsc, qsc.T) / self.temperature
    #     mask = self.mask_correlated_instances(N)
    #     negative_positive_samples = sim[mask].reshape(N, -1)
    #     indices = torch.min(cof,1).indices
    #
    #     positive_samples = torch.zeros(N).to(device)
    #
    #     for i in range(N):
    #         positive_samples[i]=negative_positive_samples[i][indices[i]]
    #         negative_positive_samples[i][indices[i]] = 0.
    #
    #     _mask = negative_positive_samples.bool()
    #     negative_samples = negative_positive_samples[_mask].reshape(N,-1)
    #     cof = cof[_mask].reshape(N, -1)
    #     cof = torch.log(cof)
    #     positive_samples = positive_samples.reshape(N,1)
    #     negative_samples = negative_samples+cof
    #
    #     labels = torch.zeros(N).to(positive_samples.device).long()
    #     logits = torch.cat((positive_samples, negative_samples), dim=1)
    #
    #     loss = self.criterion(logits, labels)
    #     loss /= N
    #
    #     return loss