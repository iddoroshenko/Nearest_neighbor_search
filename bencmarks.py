import pandas as pd
import numpy as np
import sys
# import python_wrapper.build.wrapper as wpython_wrapper/build/
import python_wrapper.engineWrapper as ew
from n2 import HnswIndex
import hnswlib
import nmslib
import time
from rpforest import RPForest
import faiss
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

def encode(d):
    return ["%s=%s" % (a, b) for (a, b) in d.items()]


if __name__ == "__main__":
    k = 100

    # Mine
    xb, xq, gt = load_sift1M()
    X = xb
    M_vec = [96]
    _k = [800]
    for M in M_vec:
        break
        start_graph = time.time()
        p = hnswlib.Index(space='l2', dim=128)
        p.init_index(max_elements=len(xb), ef_construction=500, M=M)
        p.add_items(xb)

        end_graph = time.time()
        for kk in _k:
            print("M:", M, "kk:", kk)
            start_query = time.time()
            accuracy = 0
            p.set_ef(kk)
            for i in range(len(xq)):
                ans = p.knn_query(xq[i], k=k)[0][0]
                for x in ans:
                    if x in gt[i]:
                        accuracy += 1

            end_query = time.time()
            print("M:", M, "kk:", kk)
            print('time graph:', end_graph - start_graph)
            print('time query:', end_query - start_query)
            print('time full:', end_query - start_graph)
            print('accuracy:', accuracy / len(xq) / k)

    for M in range(1, 16):
        for ef_c in range(1, 21, 5):
            print("M:", M, "ef_c:", ef_c)
            rep = 1
            alg = ew.Wrapper(M, ef_c * M, k, rep)
            X = np.asarray(xb)
            start_graph = time.time()
            alg.pySetPoints(X)
            # alg.constructGraph_reverseKNN()
            alg.constructGraph()
            end_graph = time.time()
            for ef in range(1, 10):
                print("M:", M, "ef_c:", ef_c, "ef:", ef, "rep:", rep)
                alg.pySetEf(ef*k)
                fq = open('fq_new.txt', 'a')
                start_query = time.time()
                accuracy = 0
                for i in range(len(xq)):
                    ans = alg.pyFindKNearestNeighbors(np.asarray([xq[i]]), k)
                    for x in ans:
                        if x in gt[i]:
                            accuracy += 1
                end_query = time.time()
                print("M:", M, "ef_c:", ef_c, "ef:", ef, "rep:", rep)
                print('time graph:', end_graph - start_graph)
                print('time query:', end_query - start_query)
                print('time full:', end_query - start_graph)
                print('accuracy:', accuracy / len(xq) / k)
                print(round(accuracy / len(xq) / k, 4), ": ", round(end_query - start_query, 4), ",", file=fq, sep="")
                f.close()
                fq.close()
    quit()
    arg_groups = [{"M": 32, "post": 2, "efConstruction": 400}, {"M": 20, "post": 2, "efConstruction": 400},
                  {"M": 20, "post": 2, "efConstruction": 400}, {"M": 12, "post": 0, "efConstruction": 400},
                  {"M": 4, "post": 0, "efConstruction": 400},  {"M": 8, "post": 0, "efConstruction": 400}]
    query_args = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 300, 400]
    for M in arg_groups:
        break
        index = nmslib.init(method='hnsw', space='l2')
        index.addDataPointBatch(xb)
        index.createIndex(print_progress=True)
        index_param = encode(M)
        query_param = None
        index.createIndex(index_param)
        for kk in query_args:
            print("M:", M, "kk:", kk)
            index.setQueryTimeParams(["efSearch=" + str(kk)])

            start_query = time.time()
            accuracy = 0
            for i in range(len(xq)):
                ans = index.knnQuery([xq[i]], k=k)[0]
                for x in ans:
                    if x in gt[i]:
                        accuracy += 1

            end_query = time.time()

            fq = open('nsw_fq.txt', 'a')

            print("M:", M, "kk:", kk, "acc:", round(accuracy / len(xq) / k, 4))
            print(round(accuracy / len(xq) / k, 4), ": ", round(end_query - start_query, 4), ",", file=fq, sep="")
            fq.close()
    #quit()

    args = [32,64,128,256,512,1024,2048,4096,8192]
    query = [1, 5, 10, 50, 100, 200]

    for n_list in args:
        fq = open('fq_faiss.txt', 'a')
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(quantizer, X.shape[1], n_list, faiss.METRIC_L2)
        index.train(X)
        index.add(X)
        for n_probe in query:
            fq = open('fq_faiss.txt', 'a')
            start_query = time.time()
            faiss.cvar.indexIVF_stats.reset()
            index.nprobe = n_probe
            accuracy = 0
            for i in range(len(xq)):
                v = xq[i]
                if v.dtype != np.double:
                    v = np.array(v).astype(np.double)
                ans = index.search(np.expand_dims(
                    v, axis=0).astype(np.float32), k)[1][0]
                for x in ans:
                    if x in gt[i]:
                        accuracy += 1

            end_query = time.time()
            print(n_list, n_probe)
            print(round(accuracy / len(xq) / k, 4), ": ", round(end_query - start_query, 4), ",", file=fq,
                  sep="")
            fq.close()

            print('time query:', end_query - start_query)
            print('accuracy:', accuracy / len(xq) / k)

    quit()
    a = [350]
    b = [350]

    for leaf_size in a:
        for no_trees in b:
            fq = open('fq_RPForest.txt', 'a')
            if X.dtype != np.double:
                X = np.array(X).astype(np.double)
            t = RPForest(leaf_size, no_trees)
            t.fit(X)
            start_query = time.time()
            accuracy = 0
            for i in range(len(xq)):
                v = xq[i]
                if v.dtype != np.double:
                    v = np.array(v).astype(np.double)
                ans = t.query(v, k)
                for x in ans:
                    if x in gt[i]:
                        accuracy += 1

            end_query = time.time()
            print(leaf_size, no_trees)
            print(round(accuracy / len(xq) / k, 4), ": ", round(end_query - start_query, 4), ",", file=fq,
                  sep="")
            fq.close()

            print('time query:', end_query - start_query)
            print('accuracy:', accuracy / len(xq) / k)
    # f = open('results.txt', 'a')
    # fg = open('fg.txt', 'a')
    # fq = open('fq.txt', 'a')
    # ff = open('ff.txt', 'a')

    # annoy
    f = open('annoy_results.txt', 'a')
    fg = open('annoy_fg.txt', 'a')
    fq = open('annoy_fq.txt', 'a')
    ff = open('annoy_ff.txt', 'a')
    tree_size = [100, 200, 400]
    _k = [100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000,
          100000, 200000, 400000]
    for ts in tree_size:
        for kk in _k:
            break
            print("tree_Size:", ts, "kk:", kk)
            start_graph = time.time()
            t = AnnoyIndex(128, 'euclidean')  # Length of item vector that will be indexed
            for i in range(len(xb)):
                t.add_item(i, xb[i])
            t.build(ts)  # 10 trees
            end_graph = time.time()
            start_query = end_graph
            accuracy = 0
            for i in range(len(xq)):
                ans = t.get_nns_by_vector(xq[i], k, kk)
                for x in ans:
                    if x in gt[i]:
                        accuracy += 1

            end_query = time.time()

            f = open('annoy_results.txt', 'a')
            fg = open('annoy_fg.txt', 'a')
            fq = open('annoy_fq.txt', 'a')
            ff = open('annoy_ff.txt', 'a')

            print("tree_Size:", ts, "kk:", kk, file=f)
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
    # N2

    f = open('n2_results.txt', 'a')
    fg = open('n2_fg.txt', 'a')
    fq = open('n2_fq.txt', 'a')
    ff = open('n2_ff.txt', 'a')
    M_vec = [4, 8, 12, 16, 24, 36, 48, 64, 96]
    _k = [10, 20, 40, 80, 120, 200, 400, 600, 800]
    for M in M_vec:
        break
        start_graph = time.time()
        t = HnswIndex(128)
        for i in range(len(xb)):
            t.add_data(xb[i])
        t.build(m=M, ef_construction=500)
        end_graph = time.time()
        for kk in _k:
            print("M:", M, "kk:", kk)

            start_query = time.time()
            accuracy = 0
            for i in range(len(xq)):
                ans = t.search_by_vector(xq[i], k, kk)
                for x in ans:
                    if x in gt[i]:
                        accuracy += 1

            end_query = time.time()

            f = open('n2_results.txt', 'a')
            fg = open('n2_fg.txt', 'a')
            fq = open('n2_fq.txt', 'a')
            ff = open('n2_ff.txt', 'a')

            print("M:", M, "kk:", kk, file=f)
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
    #quit()
    # HNSW

    M_vec = [4, 8, 12, 16, 24, 36, 48, 64, 96]
    _k = [10, 20, 40, 80, 120, 200, 400, 600, 800]
    for M in M_vec:
        start_graph = time.time()
        p = hnswlib.Index(space='l2', dim=128)
        p.init_index(max_elements=len(xb), ef_construction=500, M=M)
        p.add_items(xb)

        end_graph = time.time()
        for kk in _k:
            print("M:", M, "kk:", kk)
            start_query = time.time()
            accuracy = 0
            p.set_ef(kk)
            for i in range(len(xq)):
                break
                ans = p.knn_query(xq[i], k=k)[0][0]
                for x in ans:
                    if x in gt[i]:
                        accuracy += 1

            end_query = time.time()

            f = open('hnsw_results.txt', 'a')
            fg = open('hnsw_fg.txt', 'a')
            fq = open('hnsw_fq.txt', 'a')
            ff = open('hnsw_ff.txt', 'a')

            print("M:", M, "kk:", kk, file=f)
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

