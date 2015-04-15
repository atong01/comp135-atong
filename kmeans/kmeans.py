# kmeans.py
#
# Created by Alex Tong on March 04 2015
# Updated by Alex Tong on March 10 2015
#
# This program implements a basic k-means finding algorithm on .arff files.
# 
# Usage for this program:
# python kmeans.py [path_to_test_file]
#
# note that since provide only allows one file, file test.py is concatenated
# on the bottom of this file, it was used to generate test data for the
# provided test file, see results in it's header

import sys
import random
import csv
from operator import add, div

#only comes into effect when this file is used as a library, i.e. for testing of
#many values of k
RUN_EPSILON = 0.005

#parses .arff files. produces three public objects, dictdata, rawdata, and attributes.
class parser:
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

class kmeans:
    #data is represented as a list where row[0] is the associated mean index
    #means can either be a list, specifying specific starting means, or an int,
    #specifying number of means or k
    def __init__(self, rawdata, means, distfun):
        self.data = []
        self.distfun = distfun
        self.data = [[0] + row for row in rawdata]
        if type(means) is int:
            self.k = means
            self.means = self.initmeans()
        else:
            self.means = means
            self.k = len(means)
        self.dims = len(self.data[0][1:])
    #associates each data point with the nearest mean
    def reassociate(self):
        for row in self.data:
            d = self.distfun(row[1:], self.means[row[0]])
            for i, mean in enumerate(self.means):
                dp = self.distfun(row[1:], mean)
                row[0] = i if dp < d else row[0]
    #recaculates means based on mean of associated datapoints
    #if no associated datapoints, then assigns to a random datapoint
    def remean(self):
        #chooses a random new point from clusters of size greater than 1
        def randmean():
            counts = [0] * len(self.data)
            for r in self.data:
                counts[r[0]] += 1
            #return random.choice([x for x in self.data if counts[x[0]] > 1])
            return random.choice(filter(lambda x: counts[x[0]] > 1, self.data))
        for i in range(self.k):
            cluster = [r[1:] for r in self.data if r[0] is i]
            if len(cluster) is 0:
                self.means[i] = randmean()
            else:
                self.means[i] = [x / len(cluster) for x in reduce(lambda xs, ys: map(add, xs, ys), cluster)]
    #initializes means to new points in the dataset
    def initmeans(self):
        return [x[1:] for x in random.sample(self.data, self.k)]
    #calculates cost function by the sum of the squared distances of each point
    #to it's cluster center
    def quality(self):
        return sum([self.distfun(self.means[r[0]], r[1:]) for r in self.data])

#xs,ys are real valued lists representing vectors
def squaredCartesianDist(xs, ys):
    return sum([(x-y)**2 for x,y in zip(xs, ys)])
def getInt(q):
    x = None
    while not x:
        try:
            x = int(raw_input(q))
        except ValueError:
            print >> sys.stderr, "Invalid Number"
    return x
def openAny(f, op):
    try:
        return open(f, op)
    except:
        print >> sys.stderr, "file failed to open"
        sys.exit(1)
#case insensitive
def ynQuery(q):
    yes = set(['yes', 'y'])
    no  = set(['no', 'n'])
    while True:
        c = raw_input(q).lower()
        if c in yes:
            return True
        elif c in no:
            return False
        else:
            print "Please respond with 'yes' or 'no'"
#k defaults to none, if k is passed, then skips prompt for k and runs with
#   that k value
def run(k = None, RUN_FOREVER = False):
    p = parser(openAny(sys.argv[-1], "r"))
    #Toggle to skip run again
    while True:
        if k is None:
            k = getInt("k? ")
        if k <= 0:
            print "k <= 0 is not applicable"
            sys.exit(1)
        obj = kmeans(p.rawdata, k, squaredCartesianDist)
        #keeps a list of all quality function values, currently unused
        qs = [0.0]
        while True:
            obj.reassociate()
            qs.append(obj.quality())
            print obj.means
            print qs[-1]
            if RUN_FOREVER and ((qs[-1] - qs[-2])**2 < RUN_EPSILON):
                return qs[-1]
            if (not RUN_FOREVER) and not ynQuery("run again? "):
                k = None
                break
            obj.remean()

if __name__ == "__main__":
    run()
###############################################################################
# END OF FILE
###############################################################################

###############################################################################
# BEGIN OF FILE testkmeans.py
###############################################################################

#testkmeans.py
#
#Created by Alex Tong on March 10 2015
#Updated by Alex Tong on March 10 2015
#
#Test data aquired as the min quality value for each k run over 20 iterations.
#The command used to generate this data is:
#    python testkmeans.py [testfile]
# 1 : 1273351.38
# 2 : 836432.84081
# 3 : 560684.908501
# 4 : 341667.849507
# 5 : 123543.086958
# 6 : 120751.690999
# 7 : 117219.294254
# 8 : 117457.591973
# 9 : 115617.209181
#10 : 114539.934832
#11 : 115798.567453
#12 : 112127.495889
#13 : 112106.99329
#14 : 110669.429419
#15 : 110665.094093
#16 : 109089.097422
#17 : 108945.13713
#18 : 108062.865149
#19 : 110592.719265
#20 : 104144.036534
#21 : 103844.373696
#22 : 105858.537989
#23 : 101928.961581
#24 : 105803.434798
#25 : 106071.097392
#26 : 108282.084023
#27 : 105008.134663
#28 : 105096.342568
#29 : 102076.680087
#30 : 106594.176483

#From this data it is clear to see that after k=5 there is no significant gain
#for increasing values of k. Thus k=5 is probably the best fit for our data. 

def test(iter = 20, numk = 30):
    ks = [sys.maxint] * (numk)
    for k in range(1,numk):
        for i in range(iter):
            newk = kmeans.run(k, True)
            if newk < ks[k]:
                ks[k] = newk
    for i, k in enumerate(ks[1:], 1):
        print i, ":", k
###############################################################################
# END OF FILE
###############################################################################
