import numpy as np

def loadResultData(path: str):
    with open(path) as f:
        data = []
        while True:
            line = f.readline().strip()
            if not line:
                break
            data.append(float(line))
    return data

truedata = np.array(loadResultData("output/predict_data/label.csv"))
predicted = np.array(loadResultData("output/predict_data/predict.csv"))

print("MAE: ", np.mean(np.abs(truedata - predicted)))