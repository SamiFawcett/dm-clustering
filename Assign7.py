import numpy as np
import pandas as pd
import sys


def generateClusterLabels(clusterLabelData):

    clusterLabels = []
    for value in clusterLabelData:
        if(value <= 40):
            clusterLabels.append(0)
        elif(value <= 60):
            clusterLabels.append(1)
        elif(value <= 100):
            clusterLabels.append(2)
        else:
            clusterLabels.append(3)

    return clusterLabels


def parseFile(filename, k):
    headers = list(pd.read_csv(filename, nrows=0))
    # remove data/time and rv2 attributes
    headers.pop(len(headers) - 1)
    headers.pop(0)

    data = pd.read_csv(filename, usecols=headers).to_numpy()
    data_T = np.transpose(data)
    applianceAttributeData = data_T[0]  # appliances attribute data
    trueClusterLabels = generateClusterLabels(applianceAttributeData)


if __name__ == "__main__":

    arguments = sys.argv
    filename = arguments[1]
    k = int(arguments[2])  # num of clusters
    eps = float(arguments[3])  # convergence threshold
    ridge = float(arguments[4])
    max_iter = int(arguments[5])  # max iterations

    # parse
    parseFile(filename, k)
