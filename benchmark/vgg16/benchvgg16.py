from chainer.links.model.vision import vgg
import numpy as np
from pathos.multiprocessing import ThreadingPool as Pool



def main():
    v16 = vgg.VGG16Layers()

    #print([a])
    y = []
    pool = Pool(16)
    for i in range(10):
        #[np.load('/home/cjy/cjy/imnet/imgroup' + str(i) + '.npy').tolist() for i in range(4*i,4*i+4)])
        x = np.load('/home/cjy/cjy/imnet/imgroup' + str(i) + '.npy')
        t = [x[_:(_+1)] for _ in np.arange(100)]
        y.append(pool.map(v16.predict, t))
        #pool.close()

    result = []
    for z in y:
        for zz in z:
            x = np.array(zz.data)
            for zzz in x:
                result.append(zzz.argmax())

    print(np.array(result).shape)

    np.save('/home/cjy/cjy/result/vgg16', np.array(result, dtype=np.int32))

if __name__ == "__main__":
    main()
