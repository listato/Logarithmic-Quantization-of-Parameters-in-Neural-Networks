from chainer.links.model.vision import vgg
import numpy as np
import multiprocessing as mp

def main():
    v16 = vgg.VGG16Layers()

    y = []
    for i in range(125):
        pool = mp.Pool(4)
        y.append(pool.map(v16.predict, [np.load('E:\imagenet\imgroup' + str(i) + '.npy') for i in range(4*i,4*i+4)]))
        pool.close()
    result = []
    for z in y:
        for zz in z:
            x = np.array(zz.data)
            for zzz in x:
                result.append(zzz.argmax())

    print(np.array(result).shape)

    np.savetxt('D:\BaiduNetdiskDownload\\result\\vgg16.txt', np.array(result), fmt="%u")

if __name__ == "__main__":
    main()
