"""
input: numpy array

output: numpy array

"""
import numpy
import cupy


class LogQuant2:
    def __init__(self,layer,bitwidth=3):
        self.layer_data = layer
        self.width = bitwidth
        self.maxima = numpy.amax(layer)
        self.minima = numpy.amin(layer)
        self.sign = numpy.sign(layer)


    def __clip(self, x):
        if(x < 1 - 2**self.width):
            return 2
        elif(x > 0):
            return 0
        else:
            return x

    def __round(self,x):
        halfprecision = (2**(numpy.ceil(x)) - 2**(numpy.floor(x)))/2
        fractional = 2**(x) - 2**(numpy.floor(x))
        if fractional >= halfprecision:
            return numpy.ceil(x)
        else:
            return numpy.floor(x)

    @property
    def log_quantized(self):
        round = numpy.vectorize(self.__round)
        clip = numpy.vectorize(self.__clip)
        # numpy.log2(0) -> -infinity == float("-inf") which will be used in clip method
        return numpy.array(clip(round(numpy.log2(numpy.abs(self.layer_data)))),dtype=numpy.int8)

    @property
    def anti_quantized(self):
        x = numpy.power(2.0, self.log_quantized)
        x = numpy.array(x,dtype=numpy.float32)
        x[abs(x) == numpy.power(2.0, 2)] = 0
        return x * self.sign
