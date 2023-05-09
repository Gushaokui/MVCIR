from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch



class UCI(Dataset):
    def __init__(self, path):
        data1 = np.load(path + 'uci.npz')['view_0'].astype(np.float32)
        data2 = np.load(path + 'uci.npz')['view_1'].astype(np.float32)
        data3 = np.load(path + 'uci.npz')['view_2'].astype(np.float32)
        data4 = np.load(path + 'uci.npz')['view_3'].astype(np.float32)
        data5 = np.load(path + 'uci.npz')['view_4'].astype(np.float32)
        data6 = np.load(path + 'uci.npz')['view_5'].astype(np.float32)
        labels = np.load(path + 'uci.npz')['labels'].reshape(2000,1)
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]), torch.from_numpy(self.x6[idx])], torch.from_numpy(self.y[idx]).float(), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NoisyMNIST(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'NoisyMNIST.mat')['trainLabel'].astype(np.int32).reshape(50000,)
        self.V1 = scipy.io.loadmat(path + 'NoisyMNIST.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'NoisyMNIST.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 50000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view
        # data = {'X1': self.view1, 'X2': self.view2, 'X3': self.view5, 'X4': self.view4, 'X5': self.view3, 'Y': self.labels}
        # scipy.io.savemat('Caltech2.mat', data)

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "UCI":
        dataset = UCI('./data/')
        dims = [240, 76, 216, 47, 64, 6]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 50000
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
