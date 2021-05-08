import pandas as pd
import numpy as np
import sys
# import python_wrapper.build.wrapper as wpython_wrapper/build/
import python_wrapper.engineWrapper as ew
from n2 import HnswIndex
import hnswlib
import nmslib
import time


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def load_sift1M():
    print("Loading sift...", end='', file=sys.stderr)
    xb = fvecs_read("siftsmall/siftsmall_base.fvecs")
    xq = fvecs_read("siftsmall/siftsmall_query.fvecs")
    gt = ivecs_read("siftsmall/siftsmall_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, gt


if __name__ == "__main__":
    # Mine

    k = 100
    xb, xq, gt = load_sift1M()
    alg = ew.Wrapper(7)
    X = np.asarray(xb)

    start = time.time()
    alg.pySetPoints(X)
    alg.constructGraph(2)
    accuracy = 0
    for i in range(len(xq)):
        ans = alg.pyFindKNearestNeighbors(np.asarray([xq[i]]), k)
        for x in ans:
            if x in gt[i]:
                accuracy += 1

    end = time.time()
    print('Mine:')
    print('time:', end - start)
    print('accuracy:', accuracy / len(xq) / k)

    # N2
    f = 40
    start = time.time()
    t = HnswIndex(128)
    for i in range(len(xb)):
        t.add_data(xb[i])
    t.build()
    accuracy = 0
    for i in range(len(xq)):
        ans = t.search_by_vector(xq[i], k)
        for x in ans:
            if x in gt[i]:
                accuracy += 1

    end = time.time()
    print('N2:')
    print('Time:', end - start)
    print('accuracy: ', accuracy / len(xq) / k)

    # HNSW
    start = time.time()
    p = hnswlib.Index(space='l2', dim=128)
    p.init_index(max_elements=len(xb), ef_construction=200, M=16)
    p.add_items(xb)

    accuracy = 0
    for i in range(len(xq)):
        ans = p.knn_query(xq[i], k=k)[0][0]
        for x in ans:
            if x in gt[i]:
                accuracy += 1

    end = time.time()
    print('HNSW:')
    print('Time:', end - start)
    print('accuracy: ', accuracy / len(xq) / k)

    # nsw
    start = time.time()
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(xb)
    index.createIndex({'post': 2}, print_progress=True)

    accuracy = 0
    for i in range(len(xq)):
        ans = index.knnQuery(xq[i], k=k)[0]
        for x in ans:
            if x in gt[i]:
                accuracy += 1

    end = time.time()
    print('NSW:')
    print('Time:', end - start)
    print('accuracy: ', accuracy / len(xq) / k)
