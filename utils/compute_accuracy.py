from chainer.dataset import concat_examples
import chainer.functions as F

class Accuracy:
    def __init__(self, iter, model):
        self.iteration = iter.next()
        self.model = model
        
    @property
    def accuracy(self):
        image_test, target_test = concat_examples(self.iteration, -1)
        prediction_test = self.model(image_test)
        accuracy = F.accuracy(prediction_test, target_test)
        return accuracy.data
