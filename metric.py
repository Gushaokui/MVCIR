from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score, calinski_harabaz_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size):
    """
    :return:
    total_pred: prediction among all modalities
    pred_vectors: predictions of each modality, list
    labels_vector: true label
    Hs: high-level features
    Zs: low-level features
    """
    model.eval()
    soft_vector = []
    soft_vector2 = []
    pred_vectors = []
    Hs = []
    Zs = []
    total_Zs = []
    for v in range(view):
        pred_vectors.append([])
        Hs.append([])
        Zs.append([])
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        ms = model.forward_mask(xs)
        with torch.no_grad():
            qs, preds, qst = model.forward_cluster(xs)
            qs2, preds, qst2 = model.forward_cluster(ms)
            _, _, zs = model.forward(xs)
            param = model.get_parpam()
            for v in range(view):
                qs[v] = qs[v]
                qs2[v] = param[v] * qs2[v]
            q = sum(qs) / view
            q2 = sum(qs2) / view
        for v in range(view):
            zs[v] = zs[v].detach()
            preds[v] = preds[v].detach()
            pred_vectors[v].extend(preds[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
        q = q.detach()
        q2 = q2.detach()
        qst = qst.detach()

        soft_vector.extend(q.cpu().detach().numpy())
        soft_vector2.extend(q2.cpu().detach().numpy())
        total_Zs.extend(qst.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    total_pred2 = np.argmax(np.array(soft_vector2), axis=1)

    for v in range(view):
        Zs[v] = np.array(Zs[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    return total_pred, pred_vectors, labels_vector, Zs, total_Zs, total_pred2


def valid(model, device, dataset, view, data_size, class_num, eval_h=True):

    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    total_pred, pred_vectors, labels_vector, low_level_vectors, total_Zs, total_pred2 = inference(test_loader, model, device, view, data_size)
    nmi, ari, acc, pur = evaluate(labels_vector, total_pred)
    nmi2, ari2, acc2, pur2 = evaluate(labels_vector, total_pred2)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    print('ACC2 = {:.4f} NMI2 = {:.4f} ARI2 = {:.4f} PUR2={:.4f}'.format(acc2, nmi2, ari2, pur2))
    return acc, nmi, pur,acc2, nmi2, pur2
