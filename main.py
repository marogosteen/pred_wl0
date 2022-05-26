# ECG (Engineering classification of geomaterials)

# 外部ライブラリ
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 自作ライブラリ
from ml.net_model import Wl0_Net
from ml import mydataset
from ml.mydataset import Wl0Dataset
from ml import saveresult


def read_dataset(path, skiprows: int, skipcols: int):
    data = []
    header = []
    with open(path) as f:
        for row_count, line in enumerate(f.readlines()):
            if row_count == 0:
                header = line.split(",")[skipcols:]

            if row_count < skiprows:
                continue

            line = line.strip().split(",")
            line = line[skipcols:]
            line = list(map(float, line))
            data.append(line)

    return data, header


print("\nrunning...\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 250
learning_rate = 0.001
train_rate = 0.7
datasetpath = "data/dataset.csv"
output_directory = "output"
saveresult.logclear(output_directory)


net = Wl0_Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

dataset, header = read_dataset(datasetpath, skiprows=1, skipcols=5)
dataset = np.array(dataset)
print("dataset shape:", dataset.shape)

traintensor, evaltensor = mydataset.splitdata(data=dataset, train_rate=0.7)
mean = torch.mean(traintensor[:, :-1], dim=0)
std = torch.std(traintensor[:, :-1], dim=0)
transform = transforms.Lambda(lambda x: (x - mean)/std)

with open(f"{output_directory}/dataset/train.csv", "w") as f:
    f.write(",".join(header))
    for line in traintensor.tolist():
        f.write(",".join(list(map(str, line))) + "\n")

with open(f"{output_directory}/dataset/test.csv", "w") as f:
    f.write(",".join(header))
    for line in evaltensor.tolist():
        f.write(",".join(list(map(str, line))) + "\n")

train_dataset = Wl0Dataset(traintensor, transform)
eval_dataset = Wl0Dataset(evaltensor, transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=len(train_dataset), shuffle=False)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=len(eval_dataset), shuffle=False)

print(
    "train size:", len(train_dataset), "eval size:", len(eval_dataset),
    f"\nmean: {mean}", f"std: {std}", sep="\n", end="\n\n")

trainloss_hist = []
evalloss_hist = []
for epoch in tqdm(range(epochs)):
    # train
    train_net = net.train()
    count_batches = len(train_dataloader)
    trainloss = 0
    for feature, truth in train_dataloader:
        trainpred = train_net(feature)
        trainloss = loss_func(trainpred, truth)

        optimizer.zero_grad()
        trainloss.backward()
        optimizer.step()

    trainloss_hist.append(trainloss.item())

    # eval
    eval_net = net.eval()
    count_batches = len(eval_dataloader)
    evalloss = 0
    pred_hist = []
    with torch.no_grad():
        for feature, truth in eval_dataloader:
            evalpred = eval_net(feature)
            evalloss += loss_func(evalpred, truth).item()

    evalloss /= count_batches
    evalloss_hist.append(evalloss)

    saveresult.saveloss(
        trainloss_hist, f"{output_directory}/loss/trainloss.csv")
    saveresult.saveloss(
        evalloss_hist, f"{output_directory}/loss/evalloss.csv")
    saveresult.savelossimage(
        trainloss_hist, evalloss_hist, f"{output_directory}/loss/image/loss.jpg")
    saveresult.drawpred(
        truth, evalpred, f"{output_directory}/predict_image/pred.jpg")

    with open(f"{output_directory}/predict_data/label.csv", mode="w") as f:
        for line in truth.tolist():
            f.write(",".join(list(map(str, line)))+"\n")

    with open(f"{output_directory}/predict_data/predict.csv", mode="w") as f:
        for line in evalpred.tolist():
            f.write(",".join(list(map(str, line)))+"\n")

torch.save(net.state_dict(), f"{output_directory}/state_dict.pt")

print("MAE", np.mean(np.abs(np.array(truth) - np.array(evalpred))))
print("RMSE", np.sqrt(min(evalloss_hist)))
print("\nDone\n")
