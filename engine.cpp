//
// Created by ilya on 09.01.2021.
//

#include "engine.h"

uint32_t Edge::maxAge = 1;
uint32_t Edge::edgeNextId = 1;

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
}

Points AlgorithmKNN::getPoints() const {
    return points;
}

void AlgorithmKNN::constructGraph_Naive() {
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
        for(const std::pair<int,int>& x : nearestK) {
            graph[i].push_back(Edge(x.second));
            id++;
        }
    }
}

void AlgorithmKNN::constructGraph() {
    tqdm bar;
    Points allPoints = points;
    points.clear();
    for(int i = 0; i < allPoints.size(); ++i) {
        std::vector<int> knn = findKNearestNeighborsMultiStart(allPoints[i]);
        for(int x : knn) {
            graph[allPoints[i].id].push_back(Edge(allPoints[x].id));
            graph[allPoints[x].id].push_back(Edge(allPoints[i].id));
        }
        points.push_back(allPoints[i]);
        bar.progress(i, allPoints.size());
    }
}

void AlgorithmKNN::constructGraph_reverseKNN() {
    tqdm bar;
    Points allPoints = points;
    points.clear();
    for (int i = 0; i < allPoints.size(); ++i) {
        std::vector<int> knn = findKNearestNeighborsMultiStart(allPoints[i], 2*K);
        int counter = 0;
        std::set<int> newEdges;/*
        for(int j = 0; j < std::min(size_t(K), knn.size()); ++j) {
            newEdges.insert(knn[j]);
        }*/
        for (int x : knn) {
            if(isPointTheNeighbor(allPoints[i], allPoints[x])) {
                newEdges.insert(x);
                ++counter;
                break;
            }
            if(counter == K)
                break;
        }
        for(auto j : newEdges) {
            graph[allPoints[i].id].push_back(Edge(allPoints[j].id));
            graph[allPoints[j].id].push_back(Edge(allPoints[i].id));
        }
        points.push_back(allPoints[i]);
        bar.progress(i, allPoints.size());
    }
}

bool AlgorithmKNN::isPointTheNeighbor(const Point& newPoint, const Point& oldPoint, int k) {
    if (points.empty())
        return {};
    if (k == -1)
        k = K;
    int indexStartPoint = oldPoint.id;

    graph[indexStartPoint].push_back(Edge(newPoint.id));

    long double dist = distance.calculateEuclideanDistance(points[indexStartPoint], newPoint);

    std::priority_queue<std::pair<long double, int>, std::vector<std::pair<long double, int>>, std::greater<std::pair<long double, int>>> candidates;
    candidates.push({dist, indexStartPoint});

    std::unordered_map<int,int> visitedPoints;
    visitedPoints[indexStartPoint] = 1;

    std::priority_queue<std::pair<long double, int>> topKNearestPoints;

    while(!candidates.empty()) {
        std::pair<long double, int> currentPoint = candidates.top();
        candidates.pop();

        topKNearestPoints.push(currentPoint);
        if(topKNearestPoints.size() == k + 1) {
            /*if (topKNearestPoints.top().second == currentPoint.second) {
                topKNearestPoints.pop();
                break;
            }*/
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
    bool result = false;
    while(!topKNearestPoints.empty()) {
        if(topKNearestPoints.top().second == newPoint.id) {
            result = true;
            break;
        }
        topKNearestPoints.pop();
    }
    graph[indexStartPoint].pop_back();
    return result;
}


std::vector<int> AlgorithmKNN::findKNearestNeighbors(const Point& newPoint, int k) {
    if (points.empty())
        return {};
    if (k == -1)
        k = K;
    int indexStartPoint = rand() % points.size();
    long double dist = distance.calculateEuclideanDistance(points[indexStartPoint], newPoint);
    std::priority_queue<std::pair<long double, int>, std::vector<std::pair<long double, int>>, std::greater<std::pair<long double, int>>> candidates;
    candidates.push({dist, indexStartPoint});

    std::unordered_map<int,int> visitedPoints;
    visitedPoints[indexStartPoint] = 1;

    std::priority_queue<std::pair<long double, int>> topKNearestPoints;

    while(!candidates.empty()) {
        std::pair<long double, int> currentPoint = candidates.top();
        candidates.pop();

        topKNearestPoints.push(currentPoint);
        if(topKNearestPoints.size() == k + 1) {
            /*if (topKNearestPoints.top().second == currentPoint.second) {
                topKNearestPoints.pop();
                break;
            }*/
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
    result.reserve(k);
    while(!topKNearestPoints.empty()) {
        result.push_back(topKNearestPoints.top().second);
        topKNearestPoints.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<int> AlgorithmKNN::findKNearestNeighborsMultiStart(const Point& newPoint, int k) {
    if(k == -1)
        k = K;
    const int repeat = 5;
    std::unordered_map<int, int> counter;
    for(int i = 0; i < repeat; ++i) {
        std::vector<int> x = findKNearestNeighbors_Age(newPoint, k);
        for(int point : x) {
            ++counter[point];
        }
    }
    std::vector<std::pair<int,int>> t;
    t.reserve(counter.size());
    for(const std::pair<const int, int>& x : counter) {
        t.emplace_back(x.second, x.first);
    }
    std::sort(t.rbegin(), t.rend());
    std::vector<int> result(std::min(size_t(k), t.size()));
    for(int i = 0; i < std::min(size_t(k), t.size()); ++i) {
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
    std::vector<int> knn = findKNearestNeighbors(point);
    for(int x : knn) {
        graph[point.id].push_back(Edge(points[x].id));
        graph[points[x].id].push_back(Edge(point.id));
    }
    points.push_back(point);
}

uint64_t AlgorithmKNN::getCallDistanceCounter() const {
    return distance.getCallCounter();
}

std::vector<int> AlgorithmKNN::findKNearestNeighbors_Age(const Point& newPoint, int k) {
    if(k == -1)
        k = K;
    //Point nearestPoint = findOneNearestNeighborUsingAge(newPoint);
    return findKNearestNeighbors_existingPoint(newPoint, k);
}


Point AlgorithmKNN::findOneNearestNeighborUsingAge(const Point& newPoint) {
    if (points.empty())
        return {};
    int indexCurPoint = rand() % points.size();
    uint32_t curAge = 1;
    long double curDistance = distance.calculateEuclideanDistance(newPoint, points[indexCurPoint]);
    while(true) {
        bool newPointFounded = false;
        std::vector<Edge> goodEdges;
        for(const Edge& edge : graph[indexCurPoint]) {
            if(edge.age >= curAge) {
                goodEdges.push_back(edge);
            }
        }
        std::sort(goodEdges.begin(), goodEdges.end(), [](const Edge& a, const Edge& b) { return a.age < b.age; });
        for(const Edge& edge : goodEdges) {
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

std::vector<int> AlgorithmKNN::findKNearestNeighbors_existingPoint(const Point& newPoint, int k) {
    if (points.empty())
        return {};
    if (k == -1)
        k = K;
    int indexStartPoint = rand() % points.size();;
    if(!graph[newPoint.id].empty())
        indexStartPoint = graph[newPoint.id].front().dest;
    long double dist = distance.calculateEuclideanDistance(points[indexStartPoint], newPoint);
    std::priority_queue<std::pair<long double, int>, std::vector<std::pair<long double, int>>, std::greater<std::pair<long double, int>>> candidates;
    candidates.push({dist, indexStartPoint});

    std::unordered_map<int,int> visitedPoints;
    visitedPoints[indexStartPoint] = 1;

    std::priority_queue<std::pair<long double, int>> topKNearestPoints;

    while(!candidates.empty()) {
        std::pair<long double, int> currentPoint = candidates.top();
        candidates.pop();

        topKNearestPoints.push(currentPoint);
        if(topKNearestPoints.size() == k + 1) {
            /*if (topKNearestPoints.top().second == currentPoint.second) {
                topKNearestPoints.pop();
                break;
            }*/
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
    result.reserve(k);
    while(!topKNearestPoints.empty()) {
        result.push_back(topKNearestPoints.top().second);
        topKNearestPoints.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}