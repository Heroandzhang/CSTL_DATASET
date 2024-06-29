# CSTL_DATASET

# The code to process a cross spatio-temporal pathology-based lung nodule dataset

1. 1.25mm_2D_detection：
## Environment Configuration：
* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`
* Ubuntu或Cento

## Pre-training weights download address：
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

## How to use
* Using the train.py file
* When using the training script, be careful to set --data-path to the root directory where you store your “1.25mm_2D_detection” folder.


2.  1.25_or_5mm_3D_detection_mhd
