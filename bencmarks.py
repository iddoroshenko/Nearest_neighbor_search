import pandas as pd
import numpy as np
import sys
# import python_wrapper.build.wrapper as wpython_wrapper/build/
import python_wrapper.engineWrapper as ew
from n2 import HnswIndex
import hnswlib
import nmslib
import time
from annoy import AnnoyIndex


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def load_sift1M():
    print("Loading sift...", end='', file=sys.stderr)
    xb = fvecs_read("sift/sift_base.fvecs")
    xq = fvecs_read("sift/sift_query.fvecs")
    gt = ivecs_read("sift/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, gt


if __name__ == "__main__":
    k = 100

    # Mine
    xb, xq, gt = load_sift1M()
    f = open('results.txt', 'a')
    fg = open('fg.txt', 'a')
    fq = open('fq.txt', 'a')
    ff = open('ff.txt', 'a')
    for M in range(5, 11):
        for ef_c in range(1, 7):
            for ef in range(1, 7):
                for rep in range(1, 3):
                    print("M:", M, "ef_c:", ef_c, "ef:", ef, "rep:", rep)
                    alg = ew.Wrapper(M, ef_c*M, ef*k, rep)
                    X = np.asarray(xb)

                    f = open('results.txt', 'a')
                    fg = open('fg.txt', 'a')
                    fq = open('fq.txt', 'a')
                    ff = open('ff.txt', 'a')
                    start_graph = time.time()

                    alg.pySetPoints(X)
                    #alg.constructGraph_reverseKNN()
                    alg.constructGraph()
                    end_graph = time.time()
                    start_query = end_graph
                    accuracy = 0
                    for i in range(len(xq)):
                        ans = alg.pyFindKNearestNeighbors(np.asarray([xq[i]]), k)
                        for x in ans:
                            if x in gt[i]:
                                accuracy += 1
                    end_query = time.time()

                    print("M:", M, "ef_c:", ef_c, "ef:", ef, "rep:", rep, file=f)
                    print('time graph:', end_graph - start_graph, file=f)
                    print('time query:', end_query - start_query, file=f)
                    print('time full:', end_query - start_graph, file=f)
                    print('accuracy:', accuracy / len(xq) / k, file=f)
                    print(round(accuracy / len(xq) / k, 4), ": ", round(end_graph - start_graph, 4), ",", file=fg, sep="")
                    print(round(accuracy / len(xq) / k, 4), ": ", round(end_query - start_query, 4), ",", file=fq, sep="")
                    print(round(accuracy / len(xq) / k, 4), ": ", round(end_query - start_graph, 4), ",", file=ff, sep="")
                    f.close()
                    fg.close()
                    fq.close()
                    ff.close()
    quit()
    # annoy

    start_graph = time.time()
    t = AnnoyIndex(128, 'euclidean')  # Length of item vector that will be indexed
    for i in range(len(xb)):
        t.add_item(i, xb[i])
    t.build(10) # 10 trees
    end_graph = time.time()
    start_query = end_graph
    accuracy = 0
    for i in range(len(xq)):
        ans = t.get_nns_by_vector(xq[i], k, 10000)
        for x in ans:
            if x in gt[i]:
                accuracy += 1

    end_query = time.time()
    print('annoy:')
    print('time graph:', end_graph - start_graph)
    print('time query:', end_query - start_query)
    print('time full:', end_query - start_graph)
    print('accuracy: ', accuracy / len(xq) / k)
    # N2

    start_graph = time.time()
    t = HnswIndex(128)
    for i in range(len(xb)):
        t.add_data(xb[i])
    t.build()
    end_graph = time.time()
    start_query = end_graph
    accuracy = 0
    for i in range(len(xq)):
        ans = t.search_by_vector(xq[i], k)
        for x in ans:
            if x in gt[i]:
                accuracy += 1

    end_query = time.time()
    print('N2:')
    print('time graph:', end_graph - start_graph)
    print('time query:', end_query - start_query)
    print('time full:', end_query - start_graph)
    print('accuracy: ', accuracy / len(xq) / k)

    # HNSW
    start_graph = time.time()
    p = hnswlib.Index(space='l2', dim=128)
    p.init_index(max_elements=len(xb), ef_construction=200, M=16)
    p.add_items(xb)

    end_graph = time.time()
    start_query = end_graph
    accuracy = 0
    for i in range(len(xq)):
        ans = p.knn_query(xq[i], k=k)[0][0]
        for x in ans:
            if x in gt[i]:
                accuracy += 1

    end_query = time.time()
    print('HNSW:')
    print('time graph:', end_graph - start_graph)
    print('time query:', end_query - start_query)
    print('time full:', end_query - start_graph)
    print('accuracy: ', accuracy / len(xq) / k)

    # nsw

    start_graph = time.time()
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(xb)
    index.createIndex({'post': 2}, print_progress=True)

    end_graph = time.time()
    start_query = end_graph
    accuracy = 0
    for i in range(len(xq)):
        ans = index.knnQuery(xq[i], k=k)[0]
        for x in ans:
            if x in gt[i]:
                accuracy += 1

    end_query = time.time()
    print('NSW:')
    print('time graph:', end_graph - start_graph)
    print('time query:', end_query - start_query)
    print('time full:', end_query - start_graph)
    print('accuracy: ', accuracy / len(xq) / k)
