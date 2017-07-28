#!/usr/bin/env bash
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -lr 0.2 -log lr02.log -sf lr02.pth.tar
sleep 10m
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -lr 0.3 -log lr03.log -sf lr03.pth.tar
sleep 10m
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -lr 0.4 -log lr04.log -sf lr04.pth.tar
sleep 10m
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -lr 0.5 -log lr05.log -sf lr05.pth.tar
sleep 10m
