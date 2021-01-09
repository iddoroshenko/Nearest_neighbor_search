//
// Created by ilya on 07.01.2021.
//

#ifndef NEAREST_NEIGHBOR_SEARCH_ENGINE_H
#define NEAREST_NEIGHBOR_SEARCH_ENGINE_H
#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <fstream>
#include <set>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <thread>
#include <random>
#include <unordered_map>
#include <queue>
#include "tqdm.h"

struct Edge {
public:
    Edge(uint32_t _id, uint32_t _dest) : id(_id), dest(_dest), age(floor(std::log2(id))) {
        maxAge = std::max(age,maxAge);
    }
    uint32_t id;
    uint32_t dest;
    uint32_t age;

private:
    static uint32_t maxAge;
};

struct Point {
    std::vector<int> coordinates;
    uint32_t id{};

    Point() = default;
    Point(std::vector<int> _coordinates, int _id) : coordinates(std::move(_coordinates)), id(_id) {}

    const int& operator[](uint32_t index) const;
    int& operator[](uint32_t index);
};

using Points = std::vector<Point>;
using adjacency_list = std::vector<std::vector<Edge>>;

class Distance {
private:
    uint64_t callCounter = 0;

private:
    void callCounterInc();

public:
    long double calculateEuclideanDistance (const Point& point1, const Point& point2);

    uint64_t getCallCounter() const;

    void resetCallCounter();

};

class AlgorithmKNN {
private:
    Distance distance;
    adjacency_list graph;
    Points points;
    int K = 5;
public:
    AlgorithmKNN(int newK = 5) : K(newK) { graph.resize(1000000); }

    void setPoints(const Points& newPoints);

    void constructGraph_Naive();

    void constructGraph();

    Point findOneNearestNeighbor(const Point& newPoint);

    std::vector<int> findKNearestNeighbors(const Point& newPoint);

    std::vector<int> findKNearestNeighbors_Naive(const Point& newPoint);

    void addPoint(const Point& point);

    uint64_t getCallDistanceCounter() const;
};

#endif //NEAREST_NEIGHBOR_SEARCH_ENGINE_H
