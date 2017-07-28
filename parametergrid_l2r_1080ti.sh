#!/usr/bin/env bash
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -wd 1e-4 -log wd01.log -sf wd01.pth.tar
sleep 10m
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -wd 2e-4 -log wd02.log -sf wd02.pth.tar
sleep 10m
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -wd 3e-4 -log wd03.log -sf wd03.pth.tar
sleep 10m
python cifar10_classifier.py -a vgg16_bn -r False -e 400 -mb 128 -tb 128 -wd 4e-4 -log wd04.log -sf wd04.pth.tar
sleep 10m