# ACNN: a Full Resolution DCNN for Medical Image Segmentation
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICRA40945.2020.9197328-darkyellow)](https://doi.org/10.1109/ICRA40945.2020.9197328)
[![arXiv](https://img.shields.io/badge/arXiv-1901.09203-b31b1b.svg)](https://arxiv.org/abs/1901.09203)

This repository provides the python code for the [ICRA 2020 paper](https://doi.org/10.1109/ICRA40945.2020.9197328):
* Xiao-Yun Zhou, Jian-Qing Zheng, Peichao Li, Guang-Zhong Yang. **ACNN: a Full Resolution DCNN for Medical Image Segmentation**. International Conference on Robotics and Automation 2020. 

This paper proposed a method for effective atrous rate setting to achieve the largest and fully-covered receptive field with a minimum number of atrous convolutional layers. Furthermore, a new and full resolution DCNN - Atrous Convolutional Neural Network (ACNN), which incorporates cascaded atrous II-blocks, residual learning and Instance Normalization (IN) is proposed.

## Requirements
[![OS](https://img.shields.io/badge/OS-Windows%7CLinux-darkblue)]()
[![PyPI pyversions](https://img.shields.io/badge/Python-3.5-blue)](https://pypi.python.org/pypi/ansicolortags/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow->%3D1.9-lightblue)](www.tensorflow.org)
[![Numpy](https://img.shields.io/badge/Numpy-1.19.5-lightblue)](https://numpy.org)
[![Scipy](https://img.shields.io/badge/Scipy--lightblue)](https://scipy.org/)

(Other settings could be also applicable)

## Usage
The code uses pre-processed .mat files as inputs, and generates inference results also in .mat files. A sample medical image scan can be found in the *data* folder. 

Please use *train_network.py* for training and *test_network.py* for testing. Change the paths to your own data before running. The code provides ACNN, deeplab v3+ and u-net for comparison. The trained models are not provided due to the size of the file. 

* Training
```
cd $DOWNLOAD_DIR/ACNN
python train_network.py
```

* Testing
```
cd $DOWNLOAD_DIR/ACNN
python test_network.py
```

## Acknowledgement
We thank the support of NVIDIA Corporation with the donation of the Titan Xp GPU used for this research.

## Citing this work
```bibtex
@inproceedings{zhou2020acnn,
  title={Acnn: a full resolution dcnn for medical image segmentation},
  author={Zhou, Xiao-Yun and Zheng, Jian-Qing and Li, Peichao and Yang, Guang-Zhong},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8455--8461},
  year={2020},
  organization={IEEE}
}
```
