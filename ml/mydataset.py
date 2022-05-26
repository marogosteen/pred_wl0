import numpy as np
import torch
from torch.utils.data import Dataset


def splitdata(data: np.ndarray, train_rate: float):
    np.random.shuffle(data)
    train_rate = 1 - train_rate
    train_size = int(len(data) * train_rate)
    evaltensor = torch.Tensor(data[:train_size])
    traintensor = torch.Tensor(data[train_size:])
    return traintensor, evaltensor


class LLPDataset(Dataset):
    def __init__(self, tensor: torch.Tensor, transform):
        self.inputdata = tensor[:, :3]
        self.truevalue = tensor[:, -1:]
        self.transform = transform

    def __len__(self):
        return len(self.inputdata)

    def __getitem__(self, index: int):
        return self.transform(self.inputdata[index]), self.truevalue[index]

    def get_values(self):
        return torch.cat([self.inputdata, self.truevalue], dim=1)


class ECGDataset(Dataset):
    def __init__(self, tensor: torch.Tensor, transform):
        self.inputdata = tensor[:, :3]
        self.truevalue = tensor[:, 3:]
        self.transform = transform

    def __len__(self):
        return len(self.inputdata)

    def __getitem__(self, index: int):
        return self.transform(self.inputdata[index]), self.truevalue[index]

    def get_values(self):
        return torch.cat([self.inputdata, self.truevalue], dim=1)


class Wl0Dataset(Dataset):
    def __init__(self, tensor: torch.Tensor, transform):
        self.inputdata = tensor[:, :-1]
        self.truevalue = tensor[:, -1:]
        self.transform = transform

    def __len__(self):
        return len(self.inputdata)

    def __getitem__(self, index: int):
        return self.transform(self.inputdata[index]), self.truevalue[index]

    def get_values(self):
        return torch.cat([self.inputdata, self.truevalue], dim=1)
