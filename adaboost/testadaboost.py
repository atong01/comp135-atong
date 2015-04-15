# testaaboost.py
# 
# Created by Alex Tong on April 15 2015
# Updated by Alex Tong on April 15 2015
#
# adapted closely from testkmeans.py
#
# runs adaboost program a number of times with different L values, and calculates
# smallest error by running adaboost.

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
