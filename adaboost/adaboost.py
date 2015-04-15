# adaboost.py
#
# Created by Alex Tong on April 11 2015
# Updated by Alex Tong on April 15 2015
#
# This program implements a basic adaboost algorithm on .arff files
#
# Run `python adaboost.py -h` for usage information
#
# classifier defined as (attribute#, cut, dir{-1,1})
# a base learner in this adaboost algorithm selects the attribute#, cut,
#   and direction of cut to minimize error on the training data
#
# test data were is found at the top of the file testadaboost.py appended below
# 

import sys
import random
import csv
import argparse
from math import log

#parses .arff files. produces three public objects, dictdata, rawdata, and attributes.
class arffParser:
    def __init__(self, file):
        self.dictdata = []
        self.rawdata = []
        self.attributes = []
        for line in file:
            l = line.strip()
            if l.startswith("%"):
                continue
            if l.startswith("@data"):
                break
            if l.startswith("@relation"):
                continue
            if l.startswith("@attribute"):
                a = l.split()
                self.attributes.append(a[1])
        dataread = csv.reader(file)
        for values in dataread:
            rowdata = [float(f) for f in values]
            self.dictdata.append(dict(zip(self.attributes, rowdata)))
            self.rawdata.append(rowdata)
def populateTrainSet(X,R,p):
    randList = [random.random() for x in X]
    pd = [p[0]]
    tsetx = []
    tsetr = []
    for i in p[1:]:
        pd.append(pd[-1] + i)
#O(N^2) function could be reduced to O(nlgn) by sorting rand list
    for r in randList:
        for i, v in enumerate(pd):
            if v > r:
                tsetx.append(X[i])
                tsetr.append(R[i])
                break
    return (tsetx, tsetr)
def calcCut(data, bestCutLoc):
    cut = 0 
    if m[0] == 0: # classify all one way, move cut to infty
        cut = float("-inf")
    elif m[0] == len(R):
        cut = float("inf")
    else:
        cut = sum([sort[m[0]][0], sort[m[0]-1][0]]) / 2 #support vectors
    return cut
def classify(attrnum,D,R):
    # sort r by data
    sort = sorted(zip(D,R),key=lambda i: i[0])
    S = [x[1] for x in sort] 
    numpos = sum(filter(lambda x: x == 1, R))# #misclassified pos above
    numneg = len(R) - numpos
    cutl = [(0, numneg, numpos)] # #mis (index, pos above, neg above)
    for s in S:
        i = 1 if s == -1 else -1
        cutl.append((cutl[-1][0]+1, cutl[-1][1] + s, cutl[-1][2] - s))
    minup = min(cutl,key=lambda i: i[1])
    mindo = min(cutl,key=lambda i: i[2])
    direc = 1 if minup[1] < mindo[2] else -1 #also can be used as reference to enum(cut)
    m = minup if minup[1] < mindo[2] else mindo #m[0] where to cut
    err = minup[1] if minup[1] < mindo[2] else mindo[2]
    cut = calcCut(sort, m)
    return (attrnum, cut, direc, err)
def trainl(Xj):
    X = Xj[0]
    R = Xj[1]
    numAttributes = len(X[0])
    cols = [[x[i] for x in X] for i in range(numAttributes)] # list by cols
    # classifier defined as (attribute#, cut, dir{-1,1}), minimum error
    return min([classify(i,D,R) for i,D in enumerate(cols)],key=lambda i: i[-1])[:-1]
def y(l, x):  #Classify datapoint x given learner l
    return 1 if l[2]*(x[l[0]] - l[1]) > 0 else -1
def updateP(P, Bj, C): # update probability distribution
    P = map(lambda p,c: Bj*p if not c else p, P,C) # decrease prob if correct
    return [p/sum(P) for p in P] #normalize
def train(trainData, l):
    L = [] # list of learners
    Lv = [] # 1 if learner is activated else 0
    B = [] # list of accuracies
    X = trainData[0]
    R = trainData[1]
    P = [1.0 / len(X)] * len(X)
    for i in range(l):
        Xj = populateTrainSet(X,R,P)
        L.append(trainl(Xj))
        Yj = [y(L[-1], x) for x in X]
        Cj = map(lambda r,y: 1 if r != y else 0, R,Yj) # list, 1 if r != y else 0
        ej = sum([p*c for p,c in zip(P,Cj)])
        Lv.append(0 if ej > .5 else 1)
        Bj = ej / (1 - ej)
        B.append(Bj)
        if ej > 0.5: continue
        P = updateP(P, Bj, Cj)
    return zip(*(L, Lv, B)) #(L, active?, accuracy) accuracy only valid if active
def printResults(L, err):

    for i,l in enumerate(L, start=1):
        print "Learner ", i, " : on Attribute ", l[0][0], " Cutoff: ", l[0][1], 
    print "Overall Error:", err
def test(testData, L):
    D = [[y(l[0], x) for x in testData[0]] for l in L]
    Y = [sum([log(1/l[2],2)*D[j][x] for j,l in enumerate(L)]) for x in range(len(testData[0]))]
    normY = [1 if yi > 0 else -1 for yi in Y]
    C = map(lambda r,y: 1 if r != y else 0, testData[1], normY)
    err = float(sum(C))/len(C)
    return err
def parse(f):
    p = arffParser(open(f, "r"))
    attr = zip(*p.rawdata)
    return (zip(*attr[:-1]), attr[-1])
def run(trainfile, testfile, l):
    trainData = parse(trainfile)
    L = train(trainData, l)
    testData = parse(testfile)
    return test(testData, L)
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", action="store", dest="TRAINFILE", help="training datafile in .arff format")
    argparser.add_argument("-T", action="store", dest="TESTFILE", help="test datafile in .arff format")
    argparser.add_argument("-L", action="store", dest="learners", type=int, help="should supply number of learners")
    r = argparser.parse_args()
    run(r.trainfile, r.testfile, r.learners)
###############################################################################
# end of file
###############################################################################

###############################################################################
# begin of file testadaboost.py
###############################################################################

'''

# testaaboost.py
# 
# created by alex tong on april 15 2015
# updated by alex tong on april 15 2015
#
# adapted closely from testkmeans.py
#
# runs adaboost program a number of times with different l values, and calculates
# smallest error by running adaboost.
#
#
# results:
#1 : 0.234375
#2 : 0.23046875
#3 : 0.234375
#4 : 0.21875
#5 : 0.21484375
#6 : 0.20703125
#7 : 0.2109375
#8 : 0.20703125
#9 : 0.203125
#10 : 0.21484375
#11 : 0.20703125
#12 : 0.2109375
#13 : 0.20703125
#14 : 0.20703125
#15 : 0.2109375
#16 : 0.20703125
#17 : 0.2109375
#18 : 0.20703125
#19 : 0.2109375
#20 : 0.203125
#
#Table above shows L plotted against the minimal error found using that # of
#learners.
#From this table we gather that while the error gradually decreases, the best
#performing number of learners for this dataset would probably be 4, as there
#is a larger drop in error, yet it is the smallest number, as to prevent
#overfitting.

import adaboost
import sys
num = 20 + 1
iterations = 50
ks = [sys.maxint] * (num)
for k in range(1,num):
    for i in range(iterations):
        new = adaboost.run("diabetes.train.arff", "diabetes.test.arff", k)
        if new < ks[k]:
            ks[k] = new
for i, k in enumerate(ks[1:], start=1):
    print >> sys.stderr, i, ":", k

'''
###############################################################################
# END OF FILE
###############################################################################
