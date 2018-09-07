from chainer.links import GoogLeNet
import numpy as np


from PIL import Image
import glob
image_list = []
for filename in glob.glob('D:\BaiduNetdiskDownload\ILSVRC2012_img_val\*.JPEG'):
    im=Image.open(filename)
    #im = vgg.prepare(im)
    # print(im)
    image_list.append(im)

print('ok')

goo = GoogLeNet()

y = goo.predict(image_list)
y1 = np.array(y.data)
print(y1.shape)
for i in y1:
    print(np.argmax(i))
print(image_list)
print()
