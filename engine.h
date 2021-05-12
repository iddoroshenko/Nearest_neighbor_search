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
#include "tqdm.h"

struct Edge {
public:
    Edge(uint32_t _dest) : id(edgeNextId), dest(_dest), age(floor(std::log2(id))) {
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

class Distance {
private:
    uint64_t callCounter = 0;

private:
    void callCounterInc();

public:
    uint32_t calculateEuclideanDistance (const Point& point1, const Point& point2);

    uint64_t getCallCounter() const;

    void resetCallCounter();

};

class AlgorithmKNN {
protected:
    Distance distance;
    std::vector<adjacency_list> graph;
    Points points;
    int K = 5;
protected:
    bool isPointTheNeighbor(const Point& newPoint, const Point& oldPoint, int k = -1, int numOfGraph = 1);

private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> distrib;
public:
    AlgorithmKNN(int newK = 5) : K(newK) {
        gen = std::mt19937 (rd());
        distrib = std::uniform_int_distribution<>(0);
    }

    void setPoints(const Points& newPoints);

    Points getPoints() const;

    // Brute-force
    void constructGraph_Naive();

    void constructGraph(int repeat = 1);

    void constructGraph_reverseKNN(int repeat = 1);

    // Using age of edges
    Point findOneNearestNeighborUsingAge(const Point& newPoint, int numOfGraph = 1);

    std::vector<int> findKNearestNeighborsMultiStart(const Point& newPoint, int k = -1, int repeat = 1, bool age = false, int numOfGraph = 1);

    // Brute-force
    std::vector<int> findKNearestNeighbors_Naive(const Point& newPoint);

    void addPoint(const Point& point);

    uint64_t getCallDistanceCounter() const;

    std::vector<int> findKNearestNeighbors_Age(const Point& newPoint, int k = -1, bool age = false, int numOfGraph = 1);

    std::vector<int> findKNearestNeighbors(const Point& newPoint, int k = -1, int numOfGraph = 1);

    std::vector<int> findKNearestNeighbors_multiGraph(const Point& newPoint, int k = -1);
};

#endif //NEAREST_NEIGHBOR_SEARCH_ENGINE_H
