#testkmeans.py
#
#Created by Alex Tong on March 10 2015
#Updated by Alex Tong on March 10 2015


#Test data aquired as the min quality value for each k run over 20 iterations.
#The command used to generate this data is:
#    python testkmeans.py 20 30 [testfile]
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

import kmeans
import sys
def test(iter = 20, k = 30)
try:
    iterations = int(sys.argv[1])
    numk = int(sys.argv[2])+1
except ValueError:
    print >> sys.stderr, "Invalid input, usage python testkmeans.py [#iters] [#ks]"
    sys.exit(1)

ks = [sys.maxint] * (numk)
for k in range(1,numk):
    for i in range(iterations):
        newk = kmeans.run(k, True)
        if newk < ks[k]:
            ks[k] = newk
for i, k in enumerate(ks[1:], 1):
    print i, ":", k
