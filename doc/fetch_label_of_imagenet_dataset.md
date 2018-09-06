Inside chainer/examples/imagenet/train_imagenet.py:

```python
image, label = self.base[i] -> self.base = chainer.datasets.LabeledImageDataset(path, root) which goes to image_dataset.py
```
