# Logarithmic-Quantization-of-Parameters-in-Neural-Networks

One of the best off-the-shelf quantization algorithm.

**It is simple (within 20 lines of code) yet powerful (state-of-the-art accuracy after direct quantization of GoogLeNet).**

If it is useful to you, please consider citing the following paper.

> A Deep Look into Logarithmic Quantization of Model Parameters in Neural Networks, Jingyong Cai, Masashi Takemoto and Hironori Nakajo, Proceedings of The 10th International Conference on Advances in Information Technology (IAIT2018)


## Note

The quantization algorithm itself is written in Python and it should be compatible with your tools with little to no modifications.

We use [Chainer](https://chainer.org/) as our testing platform. 

## Code

The quantization kernel is [LogQuant](https://github.com/CJYLab/Logarithmic-Quantization-of-Parameters-in-Neural-Networks/blob/master/utils/logquant_v3.py).

## Intro

### What is logarithmic quantization?

![Logarithmic Quantization](/img/over_view.jpg)


### How our algorithm outperforms its counterparts?

We use decimal exponents instead of pure integers which gives lower quantization noise.

The decimail exponents might exceed the given bitwidth, therefore we use a look-up table to keep the exponents.

An overview is given in the following figure:
![DLQ](/img/logquant.jpg)

For details please refer to the paper mentioned above.
