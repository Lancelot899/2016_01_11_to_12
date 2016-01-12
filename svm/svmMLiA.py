#!/usr/bin/python
#coding = utf-8

from numpy import *

def loadDataSet(fileName):
    pf = open(fileName)
    dataList=[]
    label = []
    for line in pf.readlines():
        dataline = line.strip().replace('?', '0').split(' ')
        datalen = len(dataline)
        dataline_ = []
        label.append(float(dataline[-1]))
        for i in xrange(datalen):
            dataline_.append(float(dataline[i]))
        dataList.append(dataline_)
    dataSet = array(dataList)
    dataMaxs = dataSet.max(0)
    dataMins = dataSet.min(0)
    m, n = shape(dataSet)
    dataRanges = dataMaxs - dataMins
    dataMin = tile(dataMins, (m , 1))
    dataRange = tile(dataRanges, (m, 1))
    dataSet -= dataMin
    dataSet /= dataRange
    dataSet = mat(dataSet)
    labels = mat(label).transpose()
    m,n = shape(labels)
    labels -= mat(ones((m,n)))
    labels *= 2
    labels -= mat(ones((m,n)))
    return dataSet, labels

def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

if __name__ == '__main__':
    print loadDataSet('./horse-colic.data')
