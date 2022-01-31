import os
import shutil
import matplotlib.pyplot as plt


def logclear(output_directory: str):
    if os.path.exists(output_directory):
        shutil.rmtree(f"{output_directory}")
    os.makedirs(f"{output_directory}/loss/image/")
    os.mkdir(f"{output_directory}/dataset")
    os.mkdir(f"{output_directory}/predict_image")
    os.mkdir(f"{output_directory}/predict_data")


def saveloss(losshist: list, savepath: str):
    with open(savepath, mode="w") as f:
        f.writelines("\n".join(map(str, losshist)))


def savelossimage(trainlosshist: list, evallosshist: list, savepath: str):
    fig = plt.figure()
    ax = fig.add_subplot(
        111, title="loss", xlabel="epoch", ylabel="MSE Loss")
    epochs = range(1, len(trainlosshist)+1)
    ax.plot(epochs, trainlosshist, label="train")
    ax.plot(epochs, evallosshist, label="eval")
    ax.grid()
    ax.legend()
    plt.savefig(savepath)
    plt.close()


def drawpred(real_val, evalpred, savepath):
    fig = plt.figure()
    ax = fig.add_subplot(
        111, title="pred", ylabel="pred")
    epochs = range(1, len(real_val)+1)
    ax.plot(epochs, real_val, label="true")
    ax.plot(epochs, evalpred, label="pred")
    ax.grid()
    ax.legend()
    plt.savefig(savepath)
    plt.close()
