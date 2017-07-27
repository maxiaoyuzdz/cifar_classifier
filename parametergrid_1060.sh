#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
for i in 64 128 256
do
echo batchsize=$i
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb $i -tb $i -log bs$i.log -sf bs$i.pth.tar
sleep 10m
done