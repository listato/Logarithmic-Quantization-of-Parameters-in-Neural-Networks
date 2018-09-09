from chainer.links.model.vision import vgg
import numpy as np
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

def main():
    v16 = vgg.VGG16Layers()

    #print([a])
    y = []
    for i in range(500):
        pool = Pool(5)
        #[np.load('/home/cjy/cjy/imnet/imgroup' + str(i) + '.npy').tolist() for i in range(4*i,4*i+4)])
        x = np.load('/home/cjy/cjy/imnet/imgroup' + str(i) + '.npy')
        y.append(pool.map(v16.predict, [x[_:(_+10)] for _ in np.arange(100,step=10)]))
        pool.close()

    result = []
    for z in y:
        for zz in z:
            x = np.array(zz.data)
            for zzz in x:
                result.append(zzz.argmax())

    print(np.array(result).shape)

    np.save('/home/cjy/cjy/imnet/imgroup/vgg16', np.array(result, dtype=np.int32))

if __name__ == "__main__":
    main()
