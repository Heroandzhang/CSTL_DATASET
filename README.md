# CSTL_DATASET

# The code to process a cross spatio-temporal pathology-based lung nodule dataset

## 1. 1.25mm_2D_detection：
### Environment Configuration：
* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`
* Ubuntu或Cento

### Pre-training weights download address：
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

### How to use
* Using the train.py file
* When using the training script, be careful to set --data-path to the root directory where you store your “1.25mm_2D_detection” folder.


## 2.  1.25_or_5mm_3D_detection_mhd
### Environment Configuration：
*  Ubuntu 14.04, python 2.7, CUDA 8.0, cudnn 5.1, h5py (2.6.0), SimpleITK (0.10.0), numpy (1.11.3), nvidia-ml-py (7.352.0), matplotlib (2.0.0), scikit-image (0.12.3), scipy (0.18.1), pyparsing (2.1.4), pytorch (0.1.10+ac9245a) (anaconda is recommended)

### How to preprocess data
* For preprocessing, run prepare.py.
* data_path is the unzip raw data path for “1.25_or_5mm_3D_detection_mhd”.
* preprocess_result_path is the save path for the preprocessing.
* annos_path is the path for annotations.
* 

## 3.  1.25_or_5mm_3D_detection_mhd
### Environment Configuration：
*  Ubuntu 14.04, python 2.7, CUDA 8.0, cudnn 5.1, h5py (2.6.0), SimpleITK (0.10.0), numpy (1.11.3), nvidia-ml-py (7.352.0), matplotlib (2.0.0), scikit-image (0.12.3), scipy (0.18.1), pyparsing (2.1.4), pytorch (0.1.10+ac9245a) (anaconda is recommended)

### How to preprocess data
* For preprocessing, run prepare.py.
* data_path is the unzip raw data path for “1.25_or_5mm_3D_detection_mhd”.
* preprocess_result_path is the save path for the preprocessing.
* annos_path is the path for annotations.

## 4.  1.25mm_3D_detection_bmp
### Environment Configuration：
* Python 3.6 or higher; CUDA 10.0 or higher; PyTorch 1.2 or higher; tqdm; scipy

### How to use
* Change training configuration and data configuration in config.py, especially the path to preprocessed data.
 python train.py

## 5.  class_dataset
### How to use
* python train.py
* Change the paths of train_dataset and val_dataset in train.py


