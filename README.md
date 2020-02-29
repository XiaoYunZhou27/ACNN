# ACNN: a Full Resolution DCNN for Medical Image Segmentation
This repository provides the python code for the ICRA 2020 paper:
* Xiao-Yun Zhou, Jian-Qing Zheng, Peichao Li, Guang-Zhong Yang. **ACNN: a Full Resolution DCNN for Medical Image Segmentation**. ICRA 2020. 
This paper proposed a method for effective atrous rate setting to achieve the largest and fully-covered receptive field with a minimum number of atrous convolutional layers. Furthermore, a new and full resolution DCNN - Atrous Convolutional Neural Network (ACNN), which incorporates cascaded atrous II-blocks, residual learning and Instance Normalization (IN) is proposed.

## Requirements
* Python 3.5
* Tensorflow >= 1.9
* numpy
* scipy

## Usage
The code uses pre-processed .mat files as inputs, and generates inference results also in .mat files. A sample medical image scan can be found in the *data* folder. 

Please use *train_network.py* for training and *test_network.py* for testing. Change the paths to your own data before running. The code provides ACNN, deeplab v3+ and u-net for comparison. The trained models are not provided due to the size of the file. 

## Acknowledgement
We thank the support of NVIDIA Corporation with the donation of the Titan Xp GPU used for this research.
