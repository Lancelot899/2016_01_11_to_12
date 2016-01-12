#!/usr/bin/python
#coding = utf-8

from numpy import *

def loadSimpleData():
    dataMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, labels

def loadData(fileName):
    fr = open(fileName)
    dataSet = []
    labels = []
    return dataSet, labels

def stumpClassify(dataSet, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataSet)[0], 1))
    if threshIneq == 'lt':
        retArray[dataSet[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataSet[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataSet = mat(dataArr); labels = mat(classLabels).T
    m,n = shape(dataSet)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m, 1)))
    minError = inf
    for i in xrange(n):
        rangeMin = dataSet[:,i].min(); rangeMax = dataSet[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in xrange(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataSet, i, threshVal, inequal)
            errArr = mat(ones((m, 1)))
            errArr[predictedVals == labels] = 0
            weightedErr = D.T*errArr
	    #print "split: dim : %d, thresh : %.2f, thresh ineqal: %s,\n the weighted error is %0.3f"%(i, threshVal, inequal, weightedErr)
            if weightedErr < minError:
                minError = weightedErr
                bestClasEst = predictedVals.copy()
                bestStump['dim'] = i
                bestStump['thresh'] = threshVal
                bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst 

        
def adaBoostTrainDs(dataArr, labels, numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in xrange(numIt):
        bestStump, error, classEst = buildStump(dataArr, labels, D)
        #print "D:",D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print "ClassEst:",classEst.T
        expon = multiply(-1* alpha * mat(labels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        #print "aggClassEst:", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(labels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        #print "total error:", errorRate,'\n'
        if errorRate == 0.0:break
    return weakClassArr


if __name__ == '__main__':
    dataSet, labels =  loadSimpleData()
    D = mat(ones((5, 1)) / 5)
    print adaBoostTrainDs(dataSet, labels, 9)
