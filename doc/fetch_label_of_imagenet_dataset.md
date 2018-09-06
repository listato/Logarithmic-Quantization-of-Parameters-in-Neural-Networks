Inside chainer/examples/imagenet/train_imagenet.py:

```python
image, label = self.base[i] 
|
self.base = chainer.datasets.LabeledImageDataset(path, root) 
|
image_dataset.py
```
