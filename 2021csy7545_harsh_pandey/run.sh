#!/bin/bash

echo 'Running Code'
echo $1

if [ $1 == '1' ]
then
	python DecisionTree.py $2 $3 $3 $5
elif [ $1 == '2' ]
then
	python neural_network.py $2 $3 $4
else
	echo 'Invalid Input'
fi

