from chainer.links import GoogLeNet
import numpy
import multiprocessing as mp
import glob
from PIL import Image

image_list = []
for filename in glob.glob('D:\BaiduNetdiskDownload\ILSVRC2012_img_val\*0000000*.JPEG'):
    im=Image.open(filename)
    #im = vgg.prepare(im)
    # print(im)
    image_list.append(im)
print('memory holded')


def prepare(image, size=(256, 256)):
    """Converts the given image to the numpy array for GoogLeNet.
        Note that you have to call this method before ``__call__``
    because the pre-trained GoogLeNet model requires to resize the given
    image, covert the RGB to the BGR, subtract the mean,
    and permute the dimensions before calling.
    Args:
    image (PIL.Image or numpy.ndarray): Input image.
            If an input is ``numpy.ndarray``, its shape must be
            ``(height, width)``, ``(height, width, channels)``,
            or ``(channels, height, width)``, and
            the order of the channels must be RGB.
        size (pair of ints): Size of converted images.
            If ``None``, the given image is not resized.
    Returns:
        numpy.ndarray: The converted output array.
    """
    if isinstance(image, numpy.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0, :, :]
            elif image.shape[0] == 3:
                image = image.transpose((1, 2, 0))
        image = Image.fromarray(image.astype(numpy.uint8))
    image = image.convert('RGB')
    if size:
        image = image.resize(size)
    image = numpy.asarray(image, dtype=numpy.float32)
    image = image[:, :, ::-1]
    image -= numpy.array([104.0, 117.0, 123.0], dtype=numpy.float32)  # BGR
    image = image.transpose((2, 0, 1))
    return image

image_array = []

for _ in image_list:
    x = prepare(_)
    y = x.tolist()
    array = image_array.append(y)

npzfile = numpy.array(image_array)
print(npzfile)
numpy.save('D:\BaiduNetdiskDownload\computed', npzfile)
