Inside [chainer/examples/imagenet/train_imagenet.py:](https://github.com/chainer/chainer/blob/a9982a8b426dd07eb1ec4e7695a7bc546ecc6063/examples/imagenet/train_imagenet.py#L47)

```python
image, label = self.base[i] 
|
self.base = chainer.datasets.LabeledImageDataset(path, root) 
|
image_dataset.py
```
