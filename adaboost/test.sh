#!/bin/bash
for j in `seq 1 20`;
do
    for i in `seq 1 20`;
    do
        python adaboost.py -t diabetes.train.arff -T diabetes.test.arff -L $j
    done
done
