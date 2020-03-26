# MKL-DNN Depthwise Convolution Layer - An Example

[Deep Neural Network Library (DNNL)](https://intel.github.io/mkl-dnn/) is an open-source C++ performance library for deep learning applications developed by Intel. BUT it doesn't provide a [depthwise convolution](https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/) API. Here I give an example to implement depthwise convolution by DNNL.

Please note that since DNNL is still under active updates, it may support depthwise in the future, please DO NOT rely too much on this example, please refer to the [official documentation](https://intel.github.io/mkl-dnn/dev_guide_convolution.html) and [official repository](https://github.com/intel/mkl-dnn/) FIRST.

Next, I will use Keras and DNNL separately and use the same data to implement depthwise convolution.

## Keras Implement 

Requirements:

- numpy
- tensorflow>1.80
- keras

RUN:
```
python depthwise.py
```

## DNNL Implement

Requirements(ArchLinux):
```
$ pacman -Ss dnnl
community/intel-dnnl 1.2-1 [installed]
    Intel Deep Neural Network Library (formerly mkl-dnn)
$ pacman -S intel-dnnl
```
Build:

```
g++ -std=c++11 -ldnnl depthwise.cc -o depth
```

Run:
```
./depth
```
