import numpy as np
import math


def GetInformationGain(index, X, y):
    dataElementDict = {};
    dataElementList = [];
    targetDict = {};
    targetList = [];
    entropyMatrix = np.zeros((1,1));
    elementLength = 1;
    targetLength = 1;
    for i in range(n_samples):
        if not dataElementDict.has_key(X[i][index]):
            dataElementDict[X[i][index]] = elementLength;
            dataElementList.append(X[i][index]);
            yL, xL = entropyMatrix.shape;
            entropyMatrix = np.r_[entropyMatrix, np.zeros((1, xL))];
            elementLength += 1;

        if not targetDict.has_key(y[i]):
            targetDict[y[i]] = targetLength;
            targetList.append(y[i]);
            yL, xL = entropyMatrix.shape;
            entropyMatrix = np.c_[entropyMatrix, np.zeros((yL, 1))];
            targetLength += 1;

        entropyMatrix[0,0] += 1;
        entropyMatrix[dataElementDict[X[i][index]], 0] += 1;
        entropyMatrix[0, targetDict[y[i]]] += 1;
        entropyMatrix[dataElementDict[X[i][index]], targetDict[y[i]]] += 1;

    entropyMatrix = entropyMatrix.astype(np.float);
    yL, xL = entropyMatrix.shape;

    #print dataElementList;
    #print targetList;

    result = 0.0;
    IT = 0.0;
    R = 0.0;
    PT = 0.0;
    for i in range(1, xL):
        PT = entropyMatrix[0, i]/entropyMatrix[0,0];
        if PT != 0: IT += -(PT*math.log(PT, 2));
    for j in range(1, yL):
        T = 0.0;
        PE = entropyMatrix[j, 0] / entropyMatrix[0, 0];
        for i in range(1, xL):
            P = entropyMatrix[j, i] / entropyMatrix[j, 0];
            if P != 0: T += -(P*math.log(P, 2));
        R += PE * T;
    result = IT - R;
    return result;