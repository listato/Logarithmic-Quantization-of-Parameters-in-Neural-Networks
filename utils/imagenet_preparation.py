"""
placeholder

"""
from __future__ import division
import numpy
import glob
from PIL import Image

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

def create_datasets(groups=500, imgpath='', savepath=''):
#def create_datasets(groups=500, imgpath='D:\BaiduNetdiskDownload\ILSVRC2012_img_val\\', savepath='E:\ccc\imgroup'):
    if not imgpath:
        raise ValueError
    if not savepath:
        raise ValueError
    img_lists = [[] for _ in range(groups)]
    tmp_lists = [[] for _ in range(groups)]
    for index, filename in enumerate(glob.glob(imgpath + '*.JPEG')):
                img_lists[index//(int(50000/groups))].append(filename)
    # print(numpy.array(img_lists).shape) -> (500,100)
    for index, namelists in enumerate(img_lists[0:10]):
        for jpgnames in namelists:
            im = Image.open(jpgnames)
            tmp_lists[index].append(prepare(im).tolist())
            im.close()
        numpy.save(savepath + str(index), numpy.array(tmp_lists[index], dtype=numpy.float32))
        tmp_lists[index] = []

if __name__ == "__main__":
    create_datasets()
