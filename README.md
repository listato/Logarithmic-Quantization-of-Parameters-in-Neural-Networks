# Logarithmic-Quantization-of-Parameters-in-Neural-Networks

Our logarithmic quantization algorithm is one of the best off-the-shelf quantization algorithm.

It is simple (within 20 lines of code) yet powerful (state-of-the-art accuracy after direct quantization of GoogLeNet).

If it is useful to you, please considering cite the following paper.

> A Deep Look into Logarithmic Quantization of Model Parameters in Neural Networks, Jingyong Cai, Masashi Takemoto and Hironori Nakajo, Proceedings of The 10th International Conference on Advances in Information Technology (IAIT2018)


## Note

The quantization algorithm itself is written in Python and it should be compatible to your tools with little to no modifications.

We use [Chainer](https://chainer.org/) as our testing platform. 

## Code

Quantization kernel is [LogQuant](https://github.com/CJYLab/Logarithmic-Quantization-of-Parameters-in-Neural-Networks/blob/master/utils/logquant_v3.py).

## Intro

1) What is logarithmic quantization?

![Logarithmic Quantization](/img/pro11.jpg)


2) How our algorithm outperforms its counterparts?

We use decimal exponents instead of pure integers which gives lower quantization noise.

![DLQ](/img/alg2.jpg)

For details please refer to the paper mentioned above.
