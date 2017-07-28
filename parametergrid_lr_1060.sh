#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -lr 0.6 -log lr06.log -sf lr06.pth.tar
sleep 10m
