#!/usr/bin/python
#coding = utf-8

from numpy import *
from svmMLiA import *

def smoSimple(dataSet, labels, C, toler, maxIter = 100):
    b = 0; m, n = shape(dataSet)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in xrange(m):
            fXi = float(multiply(alphas, labels).T * (dataSet * dataSet[i, :]. T)) + b
            Ei = fXi - labels[i]
            if ((labels[i] * Ei < toler) and (alphas[i] < C)) or (alphas[i] > 0):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labels).T * (dataSet * dataSet[j, :].T)) + b
                Ej = fXj - labels[j]
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labels[i] != labels[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    #print "L == H"
                    continue
                eta = 2.0 * dataSet[i,:] * dataSet[j, :].T - dataSet[j, :] * dataSet[j, :].T - dataSet[i, :] * dataSet[i, :].T
                if eta >= 0:
                    #print "eta >= 0"
                    continue
                alphas[j] -= labels[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    #print "j not moving enough"
                    continue
                alphas[i] += labels[j] * labels[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labels[i] * (alphas[i] - alphaIold) * dataSet[i, :] * dataSet[i, :].T - \
                        labels[j] * (alphas[j] - alphaJold) * dataSet[i, :] * dataSet[j, :].T
                b2 = b - Ej - labels[i] * (alphas[i] - alphaIold) * dataSet[j, :] * dataSet[j, :].T - \
                        labels[j] * (alphas[j] - alphaJold) * dataSet[j, :] * dataSet[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter : %d i : %d, pairs changed %d"%(iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number : %d"%iter
    return b, alphas





if __name__ == "__main__":
    dataSet, labels = loadDataSet('./horse-colic.data')
    b, alphas = smoSimple(dataSet, labels, 0.6, 0.001, 50)
    print b
    print alphas


