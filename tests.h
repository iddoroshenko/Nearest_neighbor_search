//
// Created by ilya on 09.01.2021.
//

#ifndef NEAREST_NEIGHBOR_SEARCH_TESTS_H
#define NEAREST_NEIGHBOR_SEARCH_TESTS_H

#include <unordered_set>
#include "engine.h"

using labeltype = size_t;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = nullptr;
    if ((fp = fopen("/proc/self/statm", "r")) == nullptr)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

void check_accuracy(Points& points) {
    AlgorithmKNN algorithmKnn_Naive;
    AlgorithmKNN algorithmKnn;
    const size_t NumPoints = 1000;
    const size_t NumPointsToCheck = 10;
    Points checkPoints;
    checkPoints.reserve(NumPoints);
    int id = 0;
    for(int i = 0; i < NumPoints; i++) {
        int randId = rand() % points.size();
        checkPoints.push_back(Point(points[randId].coordinates, id++));
    }
    algorithmKnn.setPoints(checkPoints);
    algorithmKnn.constructGraph();

    algorithmKnn_Naive.setPoints(checkPoints);
    algorithmKnn_Naive.constructGraph_Naive();

    long double accuracy = 0;
    for(int i = 0; i < NumPointsToCheck; i++) {
        int randId = rand() % checkPoints.size();
        auto v1 = algorithmKnn.findKNearestNeighborsMultiStart(points[randId]);
        auto v2 = algorithmKnn_Naive.findKNearestNeighbors_Naive(points[randId]);

        std::unordered_set<int> pp(v2.begin(), v2.end());
        for(auto x : v1) {
            if(pp.find(x) != pp.end()) {
                accuracy++;
            }
        }
    }

    std::cout << "Accuracy: " << accuracy / (5 * NumPointsToCheck) << std::endl;

}

void check_accuracy2(Points& points, const std::vector<std::vector<int>>& queries, const std::vector<std::vector<int>>& answers) {
    AlgorithmKNN algorithmKnn;
    const size_t NumPoints = points.size();
    const size_t NumPointsToCheck = queries.size();
    Points checkPoints;
    checkPoints.reserve(NumPoints);
    for(int i = 0; i < NumPoints; i++) {
        checkPoints.push_back(Point(points[i].coordinates, i));
    }
    algorithmKnn.setPoints(checkPoints);
    algorithmKnn.constructGraph();

    long double accuracy = 0;
    for(int i = 0; i < NumPointsToCheck; i++) {
        auto v1 = algorithmKnn.findKNearestNeighborsMultiStart(Point(queries[i], i));

        std::unordered_set<int> pp(answers[i].begin(), answers[i].end());
        for(auto x : v1) {
            if(pp.find(x) != pp.end()) {
                accuracy++;
            }
        }
    }

    std::cout << "Accuracy: " << accuracy / (5 * NumPointsToCheck) << std::endl;

}

int read_sift_test1B(Points& points, std::vector<std::vector<int>>& queries, std::vector<std::vector<int>>& answers) {
    int subset_size_millions = 1;

    size_t vecsize = subset_size_millions * 1000000;

    size_t qsize = 10000;
    size_t vecdim = 128;
    std::string path_q = "../bigann/bigann_query.bvecs";
    std::string path_data = "../bigann/bigann_base.bvecs";
    char path_gt[1024];
    sprintf(path_gt, "../bigann/gnd/idx_%dM.ivecs", subset_size_millions);
    auto *massb = new unsigned char[vecdim];


    std::cout << "Loading GT:\n";
    std::ifstream inputGT(path_gt, std::ios::binary);
    auto *massQA = new unsigned int[qsize * 1000];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 1000 * i), t * 4);
        if (t != 1000) {
            std::cout << "err";
            return 0;
        }
    }
    inputGT.close();
    answers.assign(qsize, std::vector<int>(5));
    for(int i = 0; i < qsize; ++i)
        for(int j = 0; j < 5; ++j)
            answers[i][j] = massQA[i*1000 + j];

    std::cout << "Loading queries:\n";
    std::ifstream inputQ(path_q, std::ios::binary);

    queries.assign(qsize, std::vector<int>(vecdim));
    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        if (in != 128) {
            std::cout << "file error";
            exit(1);
        }
        inputQ.read((char *) massb, in);
        for (int j = 0; j < vecdim; j++) {
            queries[i][j] = massb[j];
        }

    }
    inputQ.close();

    std::ifstream input(path_data, std::ios::binary);
    int in = 0;

    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

    StopW stopw = StopW();
    StopW stopw_full = StopW();

    AlgorithmKNN algorithmKnn(5);
    points.reserve(vecsize);

    int pointId = 0;

    for (int i = 0; i < vecsize; i++) {
        unsigned char mass[128];

        input.read((char *) &in, 4);
        if (in != 128) {
            std::cout << "file error";
            exit(1);
        }
        input.read((char *) massb, in);
        for (int j = 0; j < vecdim; j++) {
            mass[j] = massb[j];
        }

        points.emplace_back(Point(std::vector<int>(mass, mass + vecdim), pointId++));
        //algorithmKnn.addPoint(Point(std::vector<int>(mass, mass + vecdim), pointId++));
    }
    input.close();
    return 0;
}

int sift_test1B() {
    Points points;
    std::vector<std::vector<int>> queries;
    std::vector<std::vector<int>> answers;
    
    //std::cout << "count: " << algorithmKnn.getCallDistanceCounter() << std::endl;
    read_sift_test1B(points,queries,answers);
    //check_accuracy(points);

    check_accuracy2(points, queries,answers);

    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    return 0;

}

#endif //NEAREST_NEIGHBOR_SEARCH_TESTS_H
