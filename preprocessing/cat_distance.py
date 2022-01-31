import pandas as pd
import numpy as np

datasetpath = "data/dataset_kobe.csv"
lengthdatapath = "data/distance_kobe.csv"
depthColNum = 20
wl0ColNum = 29

datasetdf = pd.read_csv(datasetpath)
lengthdf = pd.read_csv(lengthdatapath)

newdf = datasetdf.copy()
newdf["distance"] = None

depthCol = newdf.iloc[:, depthColNum].values
wl0Col = newdf.iloc[:, wl0ColNum].values
distanceCol = newdf.iloc[:, -1].values

for distance, depth, wl0 in lengthdf.values:
    if distance == 0:
        print(distance)
        exit()
    hoge = distanceCol[(depthCol == depth) & (wl0Col == wl0)]
    if len(hoge) > 1:
        print(np.where((depthCol == depth) & (wl0Col == wl0)))
    distanceCol[(depthCol == depth) & (wl0Col == wl0)] = distance

newdf["distance"] = distanceCol

newdf.to_csv("data/new_dataset_kobe.csv")
