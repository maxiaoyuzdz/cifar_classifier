#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -wd 6e-4 -log wd06.log -sf wd06.pth.tar
sleep 10m
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -wd 7e-4 -log wd07.log -sf wd07.pth.tar
sleep 10m