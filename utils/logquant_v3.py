"""
author: CAI JINGYONG@BeatCraft Inc.

by far the best one for neural networks quantization
"""
import numpy

class LogQuant3:
    def __init__(self,layer,bitwidth=4):
        self.layer_data = layer
        self.width = bitwidth
        self.maxima = numpy.amax(layer)
        self.minima = numpy.amin(layer)
        self.sign = numpy.sign(layer)
        self.lookup = numpy.linspace(0,-7,2**self.width)

    def __round(self,x):
        idx = (numpy.abs(self.lookup - x)).argmin()
        return idx

    @property
    def log_quantized(self):
        round = numpy.vectorize(self.__round)
        return numpy.array(round(numpy.log2(numpy.abs(self.layer_data))),dtype=numpy.int8)

    @property
    def de_quantized(self):
        x = numpy.power(2.0, self.lookup[self.log_quantized])
        x = numpy.array(x,dtype=numpy.float32)
        return x * self.sign
