"""
placeholder

"""
from __future__ import division
import numpy
import glob
from PIL import Image
from chainer.links.model.vision.googlenet import prepare

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
            tmp_lists[index].append(prepare(im, size=(256,256)).tolist())
            im.close()
        numpy.save(savepath + str(index), numpy.array(tmp_lists[index], dtype=numpy.float32))
        tmp_lists[index] = []

if __name__ == "__main__":
    create_datasets()
