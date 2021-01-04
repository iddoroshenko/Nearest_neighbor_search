#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <set>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <thread>
#include <random>
#include "tqdm.h"

uint32_t maxAge = 1;

struct Edge {
    Edge(uint32_t _id, uint32_t _dest) : id(_id), dest(_dest), age(floor(std::log2(id))) {
        maxAge = std::max(age,maxAge);
    }
    uint32_t id;
    uint32_t dest;
    uint32_t age;
};

using Point = std::vector<int>;
using Points = std::vector<Point>;
using adjacency_list = std::vector<std::vector<Edge>>;
const int K = 5;

// Euclidean distance
long double calculateDistance (const Point& point1, const Point& point2) {
    uint64_t distance = 0;
    size_t size = point1.size();
    for(int i = 0; i < size; ++i) {
        distance += int64_t(point1[i] - point2[i]) * int64_t(point1[i] - point2[i]);
    }
    return sqrt(distance);
}

adjacency_list constructGraph_Naive(const Points& points) {
    using namespace std::this_thread; // sleep_for, sleep_until
    using namespace std::chrono; // nanoseconds, system_clock, seconds
    adjacency_list result(points.size());
    int id = 1;
    tqdm bar;
    for(int i = 0; i < points.size(); ++i) {
        std::set<std::pair<int,int>> nearestK;
        for(int j = 0; j < points.size(); ++j) {
            if(i == j) continue;
            nearestK.insert({calculateDistance(points[i], points[j]), j});
            if(nearestK.size() == K + 1) {
                nearestK.erase(prev(nearestK.end()));
            }
        }
        for(auto& x : nearestK) {
            result[i].push_back(Edge(id, x.second));
            id++;
        }

        sleep_for(nanoseconds(100000000));
        bar.progress(i, points.size());
    }
    return result;
}

int addNewPoint(const Point& newPoint, const Points& points, const adjacency_list& v) {
    int indexCurPoint = rand() % points.size();
    uint32_t curAge = 1;
    long double curDistance = calculateDistance(newPoint, points[indexCurPoint]);
    while(true) {
        bool newPointFounded = false;
        std::vector<Edge> goodEdges;
        for(const auto& edge : v[indexCurPoint]) {
            if(edge.age >= curAge) {
                goodEdges.push_back(edge);
            }
        }
        std::sort(goodEdges.begin(), goodEdges.end(), [](const Edge& a, const Edge& b) { return a.age < b.age; });
        for(const auto& edge : goodEdges) {
            long double dist = calculateDistance(newPoint, points[edge.dest]);
            if(dist < curDistance) {
                indexCurPoint = edge.dest;
                curDistance = dist;
                curAge = edge.age;
                newPointFounded = true;
                break;
            }
        }
        if(!newPointFounded) {
            break;
        }
    }
    return indexCurPoint;
}

std::vector<int> findKNN(const Point& newPoint, const Points& points, const adjacency_list& v) {
    int indexStartPoint = rand() % points.size();
    long double dist = calculateDistance(points[indexStartPoint], newPoint);
    std::set<std::pair<long double, int>> candidates;
    candidates.insert({dist, indexStartPoint});

    std::vector<bool> visitedPoints(points.size(), false);
    visitedPoints[indexStartPoint] = true;

    std::set<std::pair<long double, int>> topKNearestPoints;
    topKNearestPoints.insert({dist, indexStartPoint});

    while(!candidates.empty()) {
        auto currentPoint = *candidates.begin();
        candidates.erase(candidates.begin());

        topKNearestPoints.insert(currentPoint);
        if(topKNearestPoints.size() == K + 1) {
            if (prev(topKNearestPoints.end())->second == currentPoint.second) {
                topKNearestPoints.erase(prev(topKNearestPoints.end()));
                break;
            }
            topKNearestPoints.erase(prev(topKNearestPoints.end()));
        }
        for(const Edge& edge : v[currentPoint.second]) {
            if(!visitedPoints[edge.dest]) {
                visitedPoints[edge.dest] = true;
                dist = calculateDistance(points[edge.dest], newPoint);
                candidates.insert({dist, edge.dest});
            }
        }
    }
    std::vector<int> result;
    result.reserve(K);
    for(auto & x : topKNearestPoints) {
        result.push_back(x.second);
    }
    return result;
}

int main() {
    srand(time(NULL));
    auto inputFilePath = "../sample.txt";
    std::ifstream file(inputFilePath);
    if (file.is_open()) {
        std::cout << "\nReading file " << inputFilePath << '\n';
        int n;
        file >> n;
        Points v(n);
        std::cout << "n: " << n << "\n";
        for(auto& x : v) {
            x.resize(2);
            file >> x[0] >> x[1];
        }
        file.close();
        std::cout << "The file has been read" << '\n';
        /*
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);
        */
        std::cout << "\nConstructing data stucture... " << inputFilePath << '\n';
        auto graph = constructGraph_Naive(v);
        std::cout << "\nDone!\n";
        std::cout << addNewPoint(Point({1000, 395}), v, graph) + 1<< '\n';

        auto knn = findKNN(Point({1000, 395}), v, graph);
        for(auto x : knn)
            std::cout << x + 1<< " " ;

        return 0;
    } else {
        std::cout << "Unable to open file " << inputFilePath << '\n';
        return 1;
    }

}
