Inside [chainer/examples/imagenet/train_imagenet.py:](https://github.com/chainer/chainer/blob/a9982a8b426dd07eb1ec4e7695a7bc546ecc6063/examples/imagenet/train_imagenet.py#L47)

```python
image, label = self.base[i] 
|
self.base = chainer.datasets.LabeledImageDataset(path, root) 
|
image_dataset.py
```

```python
class LabeledImageDataset(dataset_mixin.DatasetMixin):

    """Dataset of image and label pairs built from a list of paths and labels.

    This dataset reads an external image file like :class:`ImageDataset`. The
    difference from :class:`ImageDataset` is that this dataset also returns a
    label integer. The paths and labels are given as either a list of pairs or
    a text file contains paths/labels pairs in distinct lines. In the latter
    case, each path and corresponding label are separated by white spaces. This
    format is same as one used in Caffe.

    .. note::
       **This dataset requires the Pillow package being installed.** In order
       to use this dataset, install Pillow (e.g. by using the command ``pip
       install Pillow``). Be careful to prepare appropriate libraries for image
       formats you want to use (e.g. libpng for PNG images, and libjpeg for JPG
       images).

    .. warning::
       **You are responsible for preprocessing the images before feeding them
       to a model.** For example, if your dataset contains both RGB and
       grayscale images, make sure that you convert them to the same format.
       Otherwise you will get errors because the input dimensions are different
       for RGB and grayscale images.

    Args:
        pairs (str or list of tuples): If it is a string, it is a path to a
            text file that contains paths to images in distinct lines. If it is
            a list of pairs, the ``i``-th element represents a pair of the path
            to the ``i``-th image and the corresponding label. In both cases,
            each path is a relative one from the root path given by another
            argument.
        root (str): Root directory to retrieve images from.
        dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.

    """

    def __init__(self, pairs, root='.', dtype=numpy.float32,
                 label_dtype=numpy.int32):
        _check_pillow_availability()
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._root = root
        self._dtype = dtype
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = _read_image_as_array(full_path, self._dtype)

        label = numpy.array(int_label, dtype=self._label_dtype)
        return _postprocess_image(image), label
```
