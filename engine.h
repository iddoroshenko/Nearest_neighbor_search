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
#include <unordered_set>
#include <queue>
#include <boost/functional/hash.hpp>
#include "tqdm.h"

struct Edge {
public:
    Edge(uint32_t _dest, uint32_t _age) : id(edgeNextId), dest(_dest), age(_age) {
        maxAge = std::max(age,maxAge);
        ++edgeNextId;
    }
    uint32_t id;
    uint32_t dest;
    uint32_t age;

private:
    static uint32_t maxAge;
    static uint32_t edgeNextId;
};

struct Point {
    std::vector<int> coordinates;
    uint32_t id{};

    Point() = default;
    Point(std::vector<int> _coordinates, int _id) : coordinates(std::move(_coordinates)), id(_id) {}

    const int& operator[](uint32_t index) const;
    int& operator[](uint32_t index);
    bool operator<(const Point& point2) const {
        return id < point2.id;
    }
};

using Points = std::vector<Point>;
using adjacency_list = std::unordered_map<uint32_t, std::vector<Edge>>;
using priority_min_queue = std::priority_queue<std::pair<uint32_t, int>, std::vector<std::pair<uint32_t, int>>, std::greater<std::pair<uint32_t, int>>>;

class Distance {
private:
    uint64_t callCounter = 0;

private:
    void callCounterInc();

public:
    uint32_t calculateEuclideanDistance (Point& point1, Point& point2);

    uint64_t getCallCounter() const;

    void resetCallCounter();

};

class AlgorithmKNN {
protected:
    Distance distance;
    adjacency_list graph;
    Points points;
    std::size_t M;
    std::size_t ef_construction;
    std::size_t rt;
protected:
    bool isPointTheNeighbor(Point& newPoint, Point& oldPoint, std::size_t k);

private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> distrib;
    std::vector<int> visitedPoints;
    std::vector<uint32_t> calculated_distance;
    int was ;
public:
    AlgorithmKNN(std::size_t newM = 5, int _ef_construction = -1, int _rt = -1) : M(newM) {
        if (_ef_construction == -1)
            ef_construction = M;
        else
            ef_construction = _ef_construction;
        if (_rt == -1)
            rt = 2;
        else
            rt = _rt;
        gen = std::mt19937 (rd());
        distrib = std::uniform_int_distribution<>(0);
    }

    void setPoints(const Points& newPoints);

    Points getPoints() const;

    // Brute-force
    void constructGraph_Naive();

    void constructGraph();

    void constructGraph_reverseKNN();

    std::vector<int> findKNearestNeighborsMultiStart(Point& newPoint, std::size_t k, std::size_t _ef, std::size_t repeat = 3);

    // Brute-force
    std::vector<int> findKNearestNeighbors_Naive(Point& newPoint);

    std::priority_queue<std::pair<uint32_t, int>> findKNearestNeighbors(Point& newPoint, std::size_t k);

    std::pair<uint32_t, int> findOneNearestNeighbors(Point& newPoint);

    int get_counter(){ return distance.getCallCounter();}

};

#endif //NEAREST_NEIGHBOR_SEARCH_ENGINE_H
