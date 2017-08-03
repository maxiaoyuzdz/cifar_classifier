#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 3e-2 -log n10.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 5e-2 -log n11.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 7e-2 -log n12.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 9e-2 -log n13.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 0.1 -log n14.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 0.3 -log n15.log
sleep 5m
python cifar100_classifier.py -a vgg16_bn -mb 1024 -tb 1024 -wd 0.5 -log n16.log
sleep 5m