# -*- coding: utf-8 -*-
from os import listdir
import operator
from numpy import *

FOLDER_DATA = "/u01/code/machine_learning/python2/data"


def classify(inX, dataSet, labels, k):
    """
    Giải thuật : 
    For every point in our dataset:
    Lấy từng điểm trong dataSet
        Tính khoảng cách giữa inX và điểm hiện tại
        sắp xếp khoảng cách theo thứ tự tăng dần
        Lấy k phần tử có khoảng cách gần nhất với inX
        Tìm ra phân loại trong cách giữa các phần tử
        Trả về kết quả phân loại như là dự báo cho phân loại inX
    """
    # lấy ra số row của dataSet
    dataSetSize = shape(dataSet)[0]

    # tính khoảng cách giữa intX và điểm hiện tại bằng cách "tính khoảng cách giữa 2 vector
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


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(FOLDER_DATA + '/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(
            FOLDER_DATA + '/trainingDigits/%s' % fileNameStr)
    testFileList = listdir(FOLDER_DATA + '/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(
            FOLDER_DATA + '/testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))


def img2vector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(lineStr[j])
    return returnVec


def main():
    testVector = img2vector(FOLDER_DATA + '/testDigits/0_13.txt')
    # print testVector
    # print testVector[0, 0:31]
    # print testVector[0, 32:61]
    handwritingClassTest()
if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
