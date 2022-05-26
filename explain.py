import os
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import random
from torchvision import transforms

from ml import net_model

SCATTER_DOT_SIZE = 4
DRAW_SPLIT_SIZE = 15
MAX_Y_AXIS_WIDTH = 0.3


def generate_y_list(x_values: list, split_size: int) -> list:
    convert_threshold = 1 / split_size
    x_value_length = len(x_values)
    x_max = max(x_values)
    x_min = min(x_values)

    sum_count = 0
    rates_pair_count_threshold = []
    rate_indexes = [None for _ in range(x_value_length)]
    for i, step in enumerate(range(1, split_size+1)):
        count_threshold = (x_max - x_min) * step / split_size + x_min

        count = 0
        for j, x in enumerate(x_values):
            if count_threshold >= x:
                count += 1

                if rate_indexes[j] is None:
                    rate_indexes[j] = i

        count -= sum_count
        sum_count += count

        rate = count / x_value_length
        rate = 0 if rate <= convert_threshold else rate
        rates_pair_count_threshold.append(rate)

    max_rate = max(rates_pair_count_threshold)
    min_rate = min(rates_pair_count_threshold)
    if max_rate != min_rate:
        for i in range(len(rates_pair_count_threshold)):
            rate = rates_pair_count_threshold[i]
            rates_pair_count_threshold[i] = (
                rate-min_rate) / (max_rate-min_rate)

    y_list = []
    for rate_index in rate_indexes:
        rate = rates_pair_count_threshold[rate_index]
        rate *= (random.random()*2-1) * MAX_Y_AXIS_WIDTH
        y_list.append(rate)

    return y_list


def read_dataset(path):
    data = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f.readlines():
            data.append(list(map(float, line.strip().split(","))))

    return data, header


train_data, feature_names = read_dataset("output/dataset/train.csv")
eval_data, _ = read_dataset("output/dataset/test.csv")

train_tensor = torch.Tensor(train_data)
eval_tensor = torch.Tensor(eval_data)
mean = torch.mean(train_tensor[:, :-1], dim=0)
std = torch.std(train_tensor[:, :-1], dim=0)
transform = transforms.Lambda(lambda x: (x - mean)/std)

model = net_model.Wl0_Net()
model.eval()
model.load_state_dict(torch.load("output/state_dict.pt"))

feature_tensor = eval_tensor[:, :-1]

explainer = shap.DeepExplainer(model, feature_tensor)
shap_values = explainer.shap_values(feature_tensor).T.tolist()

data_count = len(shap_values[0])

fig, ax = plt.subplots()
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names, rotation=35)
# ax.set_xticks(xticks)
# ax.set_xticklabels(xtick_labels)
ax.set_xlabel("SHAP")
# ax.set_xlim(xlim_min, xlim_max)

s_values = [SCATTER_DOT_SIZE for _ in range(data_count)]

for i, (feature, shap_value) in enumerate(zip(feature_tensor.T.tolist(), shap_values)):
    zip_obj = list(zip(feature, shap_value))
    random.shuffle(zip_obj)
    feature, shap_value = zip(*zip_obj)

    y_values = generate_y_list(feature, DRAW_SPLIT_SIZE)
    y_values = [y+i for y in y_values]
    color_bar = ax.scatter(shap_value, y_values, s=s_values,
                           c=feature, cmap="jet", alpha=0.75)

color_bar_ax = plt.colorbar(color_bar)
color_bar_ax.set_ticks([max(feature), min(feature)])
color_bar_ax.set_ticklabels(["high", "low"])


plt.grid()
plt.tight_layout()

if not os.path.exists("output/shap/"):
    os.mkdir("output/shap/")

plt.savefig("output/shap/shap.jpg")
plt.close()
