# -*- coding: utf-8 -*-
from numpy import *
import operator
import os.path
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    # lấy ra số row của dataSet
    dataSetSize = shape(dataSet)[0]
    # tính khoảng cách giữa intX và điểm hiện tại bằng cách "tính khoảng cách giữa 2 vector""
    # Tạo ra 1 ma trận 0 với số dòng và số cột bằng với ma trân dataSet
    # bằng cách lặp 1 ma trận [0 0] với số lần bằng kích thước của của dataSet
    # - 1
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2  # ^2
    # axis= 0 là theo cột, =1 là theo hàng với mảng 2 chiều
    sqDistances = sum(sqDiffMat, axis=1)
    distances = sqDistances**0.5  # căn bậc 2

    # trả về index của các số trong array sắp xếp theo thứ tự tăng dần
    # ví dụ [ 1.48660687  1.41421356  0.          0.1       ] thì se có kết quả là
    # [2 3 1 0] số nhỏ nhất có index là 2 có giá trị là 0, số nhỏ thứ hai có index là 3 và giá trị là 0.1
    sortedDistIndicies = argsort(distances)

    classCount = {}
    # lặp lại k phần tử gần nhất
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sort các item trong classCount theo thứ tự ngược lại
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)

    # trả về label đầu tiên
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    print dataSet
    # lấy min value của dataSet với axis=0 thì lấy theo cột. Kết quả là 1
    # array gồm 1 dòng và số cột = số cột của dataSet
    minVals = amin(dataSet, axis=0)
    print minVals
    # lấy max value của dataset với axis=0 là lây stheo cột. Kết quả là 1
    # array gồm 1 dòng và số cột = số cột của dataSet
    maxVals = amax(dataSet, axis=0)
    print maxVals
    ranges = maxVals - minVals
    print ranges

    normDataSet = zeros(shape(dataSet))
    m = shape(dataSet)[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def main():
    group, labels = createDataSet()
    # print (classify([0, 0], group, labels, 3))
    datingDataMat, datingLabels = file2matrix(
        '/u01/code/machine_learning/python2/data/datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[
               :, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # print normMat
    # print ranges
    # print minVals

    test = array([[1.0, 2], [3.0, 4.0], [5, 6], [7, 8.1]])
    # print test
    # print amin(test,axis=0)

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
