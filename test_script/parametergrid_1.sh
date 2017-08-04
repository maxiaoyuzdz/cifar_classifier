#!/usr/bin/env bash
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 8e-4 -log n3.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 1e-3 -log n4.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 3e-3 -log n5.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 5e-3 -log n6.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 7e-3 -log n7.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 9e-3 -log n8.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 1e-2 -log n9.log
sleep 5m
