//
// Created by ilya on 14.01.2021.
//

#ifndef NEAREST_NEIGHBOR_SEARCH_UNIT_TESTS_H
#define NEAREST_NEIGHBOR_SEARCH_UNIT_TESTS_H
#include <tut/tut.hpp>
#include <tut/tut_reporter.hpp>
#include "engine.h"

namespace tut
{
    struct basic{};
    typedef test_group<basic> factory;
    typedef factory::object object;
}

namespace
{
    tut::factory tf("basic test");
}

Points getData(const std::string& inputFilePath) {
    std::ifstream file(inputFilePath);
    if (file.is_open()) {
        int n;
        file >> n;
        Points points(n);
        int id = 0;
        for (auto &x : points) {
            x.coordinates.resize(2);
            file >> x[0] >> x[1];
            x.id = id++;
        }
        file.close();
        return points;
    } else {
        std::cout << "Unable to open file " << inputFilePath << '\n';
        return {};
    }
}

namespace tut
{

    template<>
    template<>
    void object::test<1>()
    {
        Points points = getData("../sample.txt");
        AlgorithmKNN algorithmKnn;
        algorithmKnn.setPoints(points);
        algorithmKnn.constructGraph();

        std::vector<int> knn = algorithmKnn.findKNearestNeighbors_multiGraph(Point({1000, 395}, 0),5);
        std::vector<int> sampleAnswer = {17, 23, 18, 19, 22};
        ensure(knn == sampleAnswer);

    }

    template<>
    template<>
    void object::test<2>()
    {
        Points points = getData("../sample.txt");
        AlgorithmKNN algorithmKnn;
        algorithmKnn.setPoints(points);
        algorithmKnn.constructGraph_reverseKNN();

        std::vector<int> knn = algorithmKnn.findKNearestNeighbors_multiGraph(Point({1000, 395}, 0),5);
        std::vector<int> sampleAnswer = {17, 23, 18, 19, 22};
        ensure(knn == sampleAnswer);

    }
}
#endif //NEAREST_NEIGHBOR_SEARCH_UNIT_TESTS_H
