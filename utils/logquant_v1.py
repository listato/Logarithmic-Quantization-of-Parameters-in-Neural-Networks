"""
Author: CAI JINGYONG @ BeatCraft, Inc & Tokyo University of Agriculture and Technology

placeholder

input: numpy array
output: numpy array
"""
import numpy

class LogQuant:
    def __init__(self,layer,bitwidth):
        self.layer_data = layer
        self.width = bitwidth
        self.maxima = numpy.amax(layer)
        self.minima = numpy.amin(layer)
        self.fsr = self.maxima - self.minima
        self.sign = numpy.sign(layer)
        pass

    def __clip(self, x):
        # min = self.fsr-(2**self.width)
        min = 4 - (2**self.width)
        if(x <= min):
            return 0
        elif(x >= 4):
            return 4 - 1
        else:
            return x

    def __round(self,x):
        bridge = numpy.sqrt(2)-1
        decimalpart, intpart = numpy.modf(x)
        if decimalpart >= bridge:
            return numpy.ceil(x)
        else:
            return numpy.floor(x)

    @property
    def log_quantized(self):
        round = numpy.vectorize(self.__round)
        clip = numpy.vectorize(self.__clip)
        # numpy.log2(0) -> -infinity == float("-inf") which will be used in clip method
        return numpy.array(clip(round(numpy.log2(abs(self.layer_data)))),dtype=numpy.int8)

    @property
    def anti_quantized(self):
        x = numpy.power(2.0, self.log_quantized)
        return x * self.sign
