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

uint32_t maxAge = 1;

struct Edge {
    Edge(uint32_t _id, uint32_t _dest) : id(_id), dest(_dest), age(floor(std::log2(id))) {
        maxAge = std::max(age,maxAge);
    }
    uint32_t id;
    uint32_t dest;
    uint32_t age;
};

struct Point{
    std::vector<int> coordinates;
    uint32_t id{};

    Point() = default;
    Point(std::vector<int> _coordinates, int _id) : coordinates(std::move(_coordinates)), id(_id) {}

    const int& operator[](uint32_t index) const {
        return coordinates[index];
    }
    int& operator[](uint32_t index) {
        return coordinates[index];
    }
};

using Points = std::vector<Point>;
using adjacency_list = std::vector<std::vector<Edge>>;
const int K = 5;

// Euclidean distance
long double calculateDistance (const Point& point1, const Point& point2) {
    uint64_t distance = 0;
    size_t size = point1.coordinates.size();
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

int findOneNearestNeighbor(const Point& newPoint, const Points& points, const adjacency_list& v) {
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

std::vector<int> findKNearestNeighbors(const Point& newPoint, const Points& points, const adjacency_list& v) {
    int indexStartPoint = rand() % points.size();
    long double dist = calculateDistance(points[indexStartPoint], newPoint);
    std::priority_queue<std::pair<long double, int>, std::vector<std::pair<long double, int>>, std::greater<std::pair<long double, int>>> candidates;
    candidates.push({dist, indexStartPoint});

    std::unordered_map<int,int> visitedPoints;
    visitedPoints[indexStartPoint] = 1;

    std::priority_queue<std::pair<long double, int>> topKNearestPoints;

    while(!candidates.empty()) {
        auto currentPoint = candidates.top();
        candidates.pop();

        topKNearestPoints.push(currentPoint);
        if(topKNearestPoints.size() == K + 1) {
            if (topKNearestPoints.top().second == currentPoint.second) {
                topKNearestPoints.pop();
                break;
            }
            topKNearestPoints.pop();
        }
        for(const Edge& edge : v[currentPoint.second]) {
            if(!visitedPoints[edge.dest]) {
                visitedPoints[edge.dest] = 1;
                dist = calculateDistance(points[edge.dest], newPoint);
                candidates.push({dist, edge.dest});
            }
        }
    }
    std::vector<int> result;
    result.reserve(K);
    while(!topKNearestPoints.empty()) {
        result.push_back(topKNearestPoints.top().second);
        topKNearestPoints.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

int main() {
    srand(time(nullptr));
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
        auto graph = constructGraph_Naive(points);
        std::cout << "\nDone!\n";
        std::cout << findOneNearestNeighbor(Point({1000, 395}, id), points, graph) + 1 << '\n';

        auto knn = findKNearestNeighbors(Point({1000, 395}, id), points, graph);
        for(auto x : knn)
            std::cout << points[x].id + 1<< " " ;

        return 0;
    } else {
        std::cout << "Unable to open file " << inputFilePath << '\n';
        return 1;
    }

}
