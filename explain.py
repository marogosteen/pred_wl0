import torch
from torchvision import transforms
import shap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ml import mydataset
from ml import net_model


def read_dataset(path, skiprows: int, skipcols: int, headerrows: int):
    data = []
    header = []
    with open(path) as f:
        for rowCount, line in enumerate(f.readlines()):
            if rowCount == headerrows:
                header = line.strip().split(",")
                header = line[skipcols:]
                print("header", header)
            if rowCount < skiprows:
                continue

            line = line.strip().split(",")
            line = line[skipcols:]
            line = list(map(float, line))
            data.append(line)

    return data, header


datasetpath = "data/dataset.csv"
data, header = read_dataset(datasetpath, skiprows=1, skipcols=5, headerrows=0)
dataset = np.array(data)
print("dataset shape:", dataset.shape)

traintensor, evaltensor = mydataset.splitdata(data=dataset, train_rate=0.7)
mean = torch.mean(traintensor[:, :-1], dim=0)
std = torch.std(traintensor[:, :-1], dim=0)
transform = transforms.Lambda(lambda x: (x - mean)/std)

model = net_model.Wl0_Net()
model.eval()
model.load_state_dict(torch.load("output/wl0/state_dict.pt"))
# inputvalue = evaltensor[:, :-1]
# truevalue = evaltensor[:, -1:]
inputvalues = evaltensor[:, :-1]
truevalues = evaltensor[:, -1:]

print(inputvalues.shape, type(inputvalues))
print(truevalues.shape)

explainer = shap.DeepExplainer(model, inputvalues)
shap_values = explainer.shap_values(inputvalues)
# shap_values = shap_values[1]
print("inputvalues shape", inputvalues.shape, "shapvalue shape", shap_values.shape)
print("shap_values ", shap_values.shape, type(shap_values))

shap.summary_plot(shap_values, inputvalues)
