#!/bin/bash

WEIGHT_DIR=semantic/ckpt/ade20k-resnet50dilated-ppm_deepsup/

mkdir -p $WEIGHT_DIR
curl -O http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
mv encoder_epoch_20.pth $WEIGHT_DIR
curl -O http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
mv decoder_epoch_20.pth $WEIGHT_DIR

