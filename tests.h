//
// Created by ilya on 09.01.2021.
//

#ifndef NEAREST_NEIGHBOR_SEARCH_TESTS_H
#define NEAREST_NEIGHBOR_SEARCH_TESTS_H

#include <unordered_set>
#include "engine.h"

int sample() {
    auto inputFilePath = "../sample.txt";
    std::ifstream file(inputFilePath);
    if (file.is_open()) {
        std::cout << "\nReading file " << inputFilePath << '\n';
        int n;
        file >> n;
        Points points(n);
        std::cout << "n: " << n << "\n";
        int id = 0;
        for(auto& x : points) {
            x.coordinates.resize(2);
            file >> x[0] >> x[1];
            x.id = id++;
        }
        file.close();
        std::cout << "The file has been read" << '\n';
        /*
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);
        */
        std::cout << "\nConstructing data structure... " << inputFilePath << '\n';
        AlgorithmKNN algorithmKnn;
        algorithmKnn.setPoints(points);
        algorithmKnn.constructGraph();
        std::cout << "\nDone!\n";
        std::cout << algorithmKnn.findOneNearestNeighbor(Point({1000, 395}, id)).id + 1 << '\n';

        auto knn = algorithmKnn.findKNearestNeighbors(Point({1000, 395}, id));
        for(auto x : knn)
            std::cout << points[x].id + 1<< " " ;
        knn = algorithmKnn.findKNearestNeighbors_Naive(Point({1000, 395}, id));
        std::cout << '\n';
        for(auto x : knn)
            std::cout << points[x].id + 1<< " " ;

        return 0;
    } else {
        std::cout << "Unable to open file " << inputFilePath << '\n';
        return 1;
    }
}

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
    const size_t NumPointsToCheck = 100;
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

int sift_test1B() {
    int subset_size_millions = 1;

    size_t vecsize = subset_size_millions * 1000000;

    size_t qsize = 10000;
    size_t vecdim = 128;
    char *path_q = "../bigann/bigann_query.bvecs";
    char *path_data = "../bigann/bigann_base.bvecs";

    auto *massb = new unsigned char[vecdim];

    std::cout << "Loading queries:\n";
    auto *massQ = new unsigned char[qsize * vecdim];
    std::ifstream inputQ(path_q, std::ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        if (in != 128) {
            std::cout << "file error";
            exit(1);
        }
        inputQ.read((char *) massb, in);
        for (int j = 0; j < vecdim; j++) {
            massQ[i * vecdim + j] = massb[j];
        }

    }
    inputQ.close();

    std::ifstream input(path_data, std::ios::binary);
    int in = 0;

    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

    StopW stopw = StopW();
    StopW stopw_full = StopW();

    AlgorithmKNN algorithmKnn(5);
    Points points;
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
    /*algorithmKnn.setPoints(points);
    algorithmKnn.constructGraph();*/
    //std::cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";

    std::vector<std::priority_queue<std::pair<int, labeltype >>> answers;
    size_t k = 1;
    std::cout << "count: " << algorithmKnn.getCallDistanceCounter() << std::endl;

    check_accuracy(points);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    return 0;

}

#endif //NEAREST_NEIGHBOR_SEARCH_TESTS_H
