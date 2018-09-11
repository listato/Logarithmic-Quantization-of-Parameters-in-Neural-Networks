from chainer.links import GoogLeNet
import numpy as np
from pathos.multiprocessing import ThreadingPool as Pool

def main():

    gn = GoogLeNet()
    y = []
    pool = Pool(3)
    for i in range(10):
        x = np.load('E:\imagenet\imgroup' + str(i) + '.npy')
        t = [x[_:(_+1)] for _ in np.arange(100)]
        print(i)
        y.append(pool.map(gn.predict, t))

    result = []
    for z in y:
        for zz in z:
            x = np.array(zz.data)
            for zzz in x:
                result.append(np.argsort(zzz)[-5:])

    print(np.array(result).shape)

    np.save('E:\ccc\\result\\gn_over_10', np.array(result, dtype=np.int32))

if __name__=='__main__':
    main()
