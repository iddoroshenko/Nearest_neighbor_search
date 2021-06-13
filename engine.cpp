//
// Created by ilya on 09.01.2021.
//

#include <iterator>
#include "engine.h"

uint32_t Edge::maxAge = 1;
uint32_t Edge::edgeNextId = 1;

const int& Point::operator[](uint32_t index) const {
    return coordinates[index];
}

int& Point::operator[](uint32_t index) {
    return coordinates[index];
}

uint32_t Distance::calculateEuclideanDistance (Point& point1, Point& point2) {
    ++callCounter;
    uint32_t distance = 0;
    std::size_t size = point1.coordinates.size()/4;
    for(std::size_t i = 0; i < size; ++i) {
        distance += (point1[i*4] - point2[i*4]) * (point1[i*4] - point2[i*4]);
        distance += (point1[i*4+1] - point2[i*4+1]) * (point1[i*4+1] - point2[i*4+1]);
        distance += (point1[i*4+2] - point2[i*4+2]) * (point1[i*4+2] - point2[i*4+2]);
        distance += (point1[i*4+3] - point2[i*4+3]) * (point1[i*4+3] - point2[i*4+3]);
    }
    return distance;
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
    for(std::size_t i = 0; i < points.size(); ++i) {
        std::set<std::pair<uint32_t, int>> nearestK;
        for(std::size_t j = 0; j < points.size(); ++j) {
            if(i == j) continue;
            nearestK.insert({distance.calculateEuclideanDistance(points[i], points[j]), points[j].id });
            if(nearestK.size() == M + 1) {
                nearestK.erase(prev(nearestK.end()));
            }
        }
        for(const std::pair<uint32_t,int>& x : nearestK) {
            graph[i].push_back(Edge(x.second,0));
            id++;
        }
    }
}

void AlgorithmKNN::constructGraph() {
    tqdm bar;
    Points allPoints = points;
    visitedPoints.resize(points.size()+1);
    calculated_distance.resize(points.size()+1);
    was = 0;
    points.clear();
    uint32_t edge_age = 0;
    for (std::size_t i = 0; i < allPoints.size(); ++i) {
        std::vector<int> knn = findKNearestNeighborsMultiStart(allPoints[i], M, ef_construction, rt);
        for (int x : knn) {
            graph[allPoints[i].id].push_back(Edge(allPoints[x].id, floor(allPoints[i].id / 1000)));
            graph[allPoints[x].id].push_back(Edge(allPoints[i].id, floor(allPoints[i].id / 1000)));
            ++edge_age;
        }
        points.push_back(allPoints[i]);
        bar.progress(int(i), int(allPoints.size()));
    }
    std::cout << std::endl << "distance func call count: " << distance.getCallCounter() << std::endl;
    distance.resetCallCounter();
    for (auto& x : graph) {
        sort(x.second.begin(), x.second.end(), [](const Edge& a, const Edge& b) { return a.age > b.age; });
    }
}

void AlgorithmKNN::constructGraph_reverseKNN() {
    tqdm bar;
    Points allPoints = points;
    visitedPoints.resize(points.size()+1);
    was = 0;
    uint32_t edge_age = 0;
    points.clear();
    for (std::size_t i = 0; i < allPoints.size(); ++i) {
        std::vector<int> knn = findKNearestNeighborsMultiStart(allPoints[i], 2*M, std::max(2*M, ef_construction), rt);
        std::size_t counter = 0;
        std::set<int> newEdges;
        for (std::size_t j = 0; j < std::min(size_t(M), knn.size()); ++j) {
            newEdges.insert(knn[j]);
        }
        for (auto j : newEdges) {
            graph[allPoints[i].id].push_back(Edge(allPoints[j].id, 0));
            graph[allPoints[j].id].push_back(Edge(allPoints[i].id, 0));
        }
        newEdges.clear();
        for (std::size_t j = std::min(size_t(M), knn.size()); j < knn.size(); ++j) {
            if (isPointTheNeighbor(allPoints[i], allPoints[knn[j]], M)) {
                newEdges.insert(knn[j]);
                ++counter;
                //break;
            }
            if (counter == M)
                break;
        }

        for (auto j : newEdges) {
            graph[allPoints[i].id].push_back(Edge(allPoints[j].id, floor(std::log2(edge_age))));
            graph[allPoints[j].id].push_back(Edge(allPoints[i].id, floor(std::log2(edge_age))));
        }
        points.push_back(allPoints[i]);
        bar.progress(int(i), int(allPoints.size()));
    }
    std::cout << std::endl << "distance func call count: " << distance.getCallCounter() << std::endl;
}

bool AlgorithmKNN::isPointTheNeighbor(Point& newPoint, Point& oldPoint, std::size_t k) {
    if (points.empty())
        return {};
    std::size_t indexStartPoint = oldPoint.id;

    graph[indexStartPoint].push_back(Edge(newPoint.id, 0));

    uint32_t dist = distance.calculateEuclideanDistance(points[indexStartPoint], newPoint);

    priority_min_queue candidates;
    candidates.push({dist, indexStartPoint});

    ++was;
    visitedPoints[indexStartPoint] = was;

    std::priority_queue<std::pair<uint32_t, std::size_t>> topKNearestPoints;

    while(!candidates.empty()) {
        std::pair<uint32_t, std::size_t> currentPoint = candidates.top();
        candidates.pop();

        topKNearestPoints.push(currentPoint);
        if(topKNearestPoints.size() == k + 1) {
            if (topKNearestPoints.top().second == currentPoint.second) {
                topKNearestPoints.pop();
                break;
            }
            topKNearestPoints.pop();
        }
        for(const Edge& edge : graph[currentPoint.second]) {
            if(visitedPoints[edge.dest] != was) {
                visitedPoints[edge.dest] = was;
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


std::vector<int> AlgorithmKNN::findKNearestNeighborsMultiStart(Point& newPoint, std::size_t k, std::size_t _ef, std::size_t repeat) {
    std::vector<int> dists;
    std::priority_queue<std::pair<uint32_t, int>> x = findKNearestNeighbors(newPoint, _ef);
    while(x.size() > k)
        x.pop();
    int n = std::min(k, x.size());
    dists.resize(n);
    for (int i = n-1; i >= 0; i--) {
        dists[i] = x.top().second;
        x.pop();
    }
    return dists;
}

std::vector<int> AlgorithmKNN::findKNearestNeighbors_Naive(Point& newPoint) {
    priority_min_queue topKNearestPoints;

    for(std::size_t i = 0; i < points.size(); ++i) {
        topKNearestPoints.push({distance.calculateEuclideanDistance(newPoint, points[i]), i});
    }

    std::vector<int> result(M);
    for(std::size_t i = 0; i < std::min(size_t(M),topKNearestPoints.size()); i++){
        result[i] = topKNearestPoints.top().second;
        topKNearestPoints.pop();
    }
    return result;
}

std::priority_queue<std::pair<uint32_t, int>> AlgorithmKNN::findKNearestNeighbors(Point& newPoint, std::size_t k) {
    if (points.empty())
        return {};
    std::pair<uint32_t, int> nearest_point = findOneNearestNeighbors(newPoint);
    std::size_t indexStartPoint = nearest_point.second;
    uint32_t dist = nearest_point.first;
    priority_min_queue candidates;
    candidates.push({dist, indexStartPoint});
    ++was;
    visitedPoints[indexStartPoint] = was;

    std::priority_queue<std::pair<uint32_t, int>> topKNearestPoints;

    while(!candidates.empty()) {
        std::pair<uint32_t, int> currentPoint = candidates.top();
        candidates.pop();

        topKNearestPoints.push(currentPoint);
        if(topKNearestPoints.size() == k + 1) {
            if (topKNearestPoints.top().second == currentPoint.second) {
                topKNearestPoints.pop();
                break;
            }
            topKNearestPoints.pop();
        }
        for(auto edge = graph[currentPoint.second].begin(); edge != graph[currentPoint.second].end(); edge++) {
            if(visitedPoints[edge->dest] != was) {
                if(visitedPoints[edge->dest] == was-1) {
                    dist = calculated_distance[edge->dest];
                } else {
                    dist = distance.calculateEuclideanDistance(points[edge->dest], newPoint);
                }
                visitedPoints[edge->dest] = was;
                candidates.push({dist, edge->dest});
            }
        }
    }
    return topKNearestPoints;
}


std::pair<uint32_t, int> AlgorithmKNN::findOneNearestNeighbors(Point& newPoint) {
    if (points.empty())
        return {};
    distance.resetCallCounter();
    std::size_t indexStartPoint = distrib(gen) % points.size();
    uint32_t dist = distance.calculateEuclideanDistance(points[indexStartPoint], newPoint);
    calculated_distance[indexStartPoint] = dist;
    std::pair<uint32_t, int> candidate;
    candidate = {dist, indexStartPoint};

    ++was;
    visitedPoints[indexStartPoint] = was;

    std::ofstream stream;
    stream.open("dist.txt", std::ios::app);
    int a = dist;

    while(true) {
        std::pair<uint32_t, int> currentPoint = candidate;
        for(auto edge = graph[currentPoint.second].rbegin(); edge != graph[currentPoint.second].rend(); edge++) {
            if(visitedPoints[edge->dest] != was) {
                visitedPoints[edge->dest] = was;
                dist = distance.calculateEuclideanDistance(points[edge->dest], newPoint);
                calculated_distance[edge->dest] = dist;
                if(dist < candidate.first) {
                    candidate = {dist, edge->dest};
                    stream << a-dist << " " << edge->age <<  std::endl;
                    a=dist;
                    break;
                }
            }
        }
        if (currentPoint.second == candidate.second)
            break;
    }
    //std::cout << "saad: " << distance.getCallCounter() << std::endl;
    stream <<  std::endl;
    distance.resetCallCounter();
    return candidate;
}
