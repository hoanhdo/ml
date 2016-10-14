from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def main():
    test = array[ 1.48660687,  1.41421356,  0.,0.1]
    print test.argsort()
if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))