#!/usr/bin/env bash
for i in 512 1024 2048 4096
do
echo batchsize=$i
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb $i -tb $i -log bs$i.log -sf bs$i.pth.tar
sleep 10m
done