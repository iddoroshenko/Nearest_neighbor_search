//
// Created by ilya on 09.01.2021.
//

#include "engine.h"

uint32_t Edge::maxAge = 1;

const int& Point::operator[](uint32_t index) const {
    return coordinates[index];
}

int& Point::operator[](uint32_t index) {
    return coordinates[index];
}

long double Distance::calculateEuclideanDistance (const Point& point1, const Point& point2) {
    callCounterInc();
    uint64_t distance = 0;
    size_t size = point1.coordinates.size();
    for(int i = 0; i < size; ++i) {
        distance += int64_t(point1[i] - point2[i]) * int64_t(point1[i] - point2[i]);
    }
    return sqrt(distance);
}

void Distance::callCounterInc() {
    ++callCounter;
}

uint64_t Distance::getCallCounter() const {
    return callCounter;
}

void Distance::resetCallCounter() {
    callCounter = 0;
}

void AlgorithmKNN::setPoints(const Points& newPoints) {
    points = newPoints;
    graph.resize(points.size());
}

Points AlgorithmKNN::getPoints() const {
    return points;
}

void AlgorithmKNN::constructGraph_Naive() {
    using namespace std::this_thread; // sleep_for, sleep_until
    using namespace std::chrono; // nanoseconds, system_clock, seconds
    graph.resize(points.size());
    int id = 1;
    tqdm bar;
    for(int i = 0; i < points.size(); ++i) {
        std::set<std::pair<int,int>> nearestK;
        for(int j = 0; j < points.size(); ++j) {
            if(i == j) continue;
            nearestK.insert({distance.calculateEuclideanDistance(points[i], points[j]), points[j].id });
            if(nearestK.size() == K + 1) {
                nearestK.erase(prev(nearestK.end()));
            }
        }
        for(auto& x : nearestK) {
            graph[i].push_back(Edge(id, x.second));
            id++;
        }

        sleep_for(nanoseconds(100000000));
        bar.progress(i, points.size());
    }
}

void AlgorithmKNN::constructGraph() {
    using namespace std::this_thread; // sleep_for, sleep_until
    using namespace std::chrono; // nanoseconds, system_clock, seconds
    tqdm bar;
    graph.resize(std::max_element(points.begin(), points.end())->id);
    int edgeId = 1;
    Points allPoints = points;
    points.clear();
    points.reserve(allPoints.size());
    for(int i = 0; i < allPoints.size(); ++i) {
        auto knn = findKNearestNeighborsMultiStart(allPoints[i]);
        for(auto x : knn) {
            graph[allPoints[i].id].push_back(Edge(edgeId, allPoints[x].id));
            graph[allPoints[x].id].push_back(Edge(edgeId, allPoints[i].id));
            edgeId++;
        }
        points.push_back(allPoints[i]);
        //sleep_for(nanoseconds(100000000));
        bar.progress(i, allPoints.size());
    }
}

Point AlgorithmKNN::findOneNearestNeighbor(const Point& newPoint) {
    int indexCurPoint = rand() % points.size();
    uint32_t curAge = 1;
    long double curDistance = distance.calculateEuclideanDistance(newPoint, points[indexCurPoint]);
    while(true) {
        bool newPointFounded = false;
        std::vector<Edge> goodEdges;
        for(const auto& edge : graph[indexCurPoint]) {
            if(edge.age >= curAge) {
                goodEdges.push_back(edge);
            }
        }
        std::sort(goodEdges.begin(), goodEdges.end(), [](const Edge& a, const Edge& b) { return a.age < b.age; });
        for(const auto& edge : goodEdges) {
            long double dist = distance.calculateEuclideanDistance(newPoint, points[edge.dest]);
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
    return points[indexCurPoint];
}

std::vector<int> AlgorithmKNN::findKNearestNeighbors(const Point& newPoint) {
    if (points.empty())
        return {};
    int indexStartPoint = rand() % points.size();
    long double dist = distance.calculateEuclideanDistance(points[indexStartPoint], newPoint);
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
        for(const Edge& edge : graph[currentPoint.second]) {
            if(!visitedPoints[edge.dest]) {
                visitedPoints[edge.dest] = 1;
                dist = distance.calculateEuclideanDistance(points[edge.dest], newPoint);
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

std::vector<int> AlgorithmKNN::findKNearestNeighborsMultiStart(const Point& newPoint) {
    const int repeat = 10;
    std::unordered_map<int, int> counter;
    for(int i = 0; i < repeat; ++i) {
        auto x = findKNearestNeighbors(newPoint);
        for(auto point : x) {
            ++counter[point];
        }
    }
    std::vector<std::pair<int,int>> t;
    t.reserve(counter.size());
    for(auto x : counter) {
        t.emplace_back(x.second, x.first);
    }
    std::sort(t.rbegin(), t.rend());
    std::vector<int> result(K);
    for(int i = 0; i < std::min(size_t(K), t.size()); ++i) {
        result[i] = t[i].second;
    }
    return result;

}

std::vector<int> AlgorithmKNN::findKNearestNeighbors_Naive(const Point& newPoint) {
    std::priority_queue<std::pair<long double, int>, std::vector<std::pair<long double, int>>,
            std::greater<std::pair<long double, int>>> topKNearestPoints;

    for(int i = 0; i < points.size(); ++i) {
        topKNearestPoints.push({distance.calculateEuclideanDistance(newPoint, points[i]), i});
    }

    std::vector<int> result(K);
    for(int i = 0; i < std::min(size_t(K),topKNearestPoints.size()); i++){
        result[i] = topKNearestPoints.top().second;
        topKNearestPoints.pop();
    }
    return result;
}

void AlgorithmKNN::addPoint(const Point& point) {
    auto knn = findKNearestNeighbors(point);
    static int edgeId = 0;
    for(auto x : knn) {
        int newSize = std::max(point.id, points[x].id);
        if(graph.size() < newSize);
            graph.reserve(newSize);
        graph[point.id].push_back(Edge(edgeId, points[x].id));
        graph[points[x].id].push_back(Edge(edgeId, point.id));
        edgeId++;
    }
    points.push_back(point);
}

uint64_t AlgorithmKNN::getCallDistanceCounter() const {
    return distance.getCallCounter();
}