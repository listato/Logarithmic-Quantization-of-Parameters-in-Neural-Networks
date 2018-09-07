from chainer.links import GoogLeNet
import numpy as np
import multiprocessing as mp
import glob
from PIL import Image

def main():
    image_list = []
    for filename in glob.glob('D:\BaiduNetdiskDownload\ILSVRC2012_img_val\*0000000*.JPEG'):
        im=Image.open(filename)
        #im = vgg.prepare(im)
        # print(im)
        image_list.append(im)
    print('ok')
    goo = GoogLeNet()
    y = goo.predict(image_list)
    y1 = np.array(y.data)
    print(y1.shape)
    y2 = np.array([])
    for i in y1:
        x = np.argmax(i)
        print(x)
        y2 = np.append(y2, x)
    print(image_list)
    #np.savetxt('D:\BaiduNetdiskDownload\computed.txt', y2 , fmt='%u')
if __name__=='__main__':
    main()
