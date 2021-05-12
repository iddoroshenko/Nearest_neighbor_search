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

uint32_t Distance::calculateEuclideanDistance (const Point& point1, const Point& point2) {
    ++callCounter;
    uint32_t distance = 0;
    std::size_t size = point1.coordinates.size();
    for(std::size_t i = 0; i < size; ++i) {
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
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
    graph.resize(1);
    for(std::size_t i = 0; i < points.size(); ++i) {
        std::set<std::pair<uint32_t, int>> nearestK;
        for(std::size_t j = 0; j < points.size(); ++j) {
            if(i == j) continue;
            nearestK.insert({distance.calculateEuclideanDistance(points[i], points[j]), points[j].id });
            if(int(nearestK.size()) == K + 1) {
                nearestK.erase(prev(nearestK.end()));
            }
        }
        for(const std::pair<uint32_t,int>& x : nearestK) {
            graph[0][i].push_back(Edge(x.second));
            id++;
        }
    }
}

void AlgorithmKNN::constructGraph(int repeat) {
    graph.resize(repeat);
    tqdm bar;
    Points allPoints = points;
    points.clear();
    for (std::size_t i = 0; i < allPoints.size(); ++i) {
        for (int r = 0; r < repeat; ++r) {
            std::vector<int> knn = findKNearestNeighborsMultiStart(allPoints[i], K, 2, false, r);
            //std::vector<int> knn = findKNearestNeighbors_multiGraph(allPoints[i], K);
            for (int x : knn) {
                graph[r][allPoints[i].id].push_back(Edge(allPoints[x].id));
                graph[r][allPoints[x].id].push_back(Edge(allPoints[i].id));
            }
        }
        points.push_back(allPoints[i]);
        bar.progress(int(i), int(allPoints.size()));
    }
    std::cout << std::endl << "distance func call count: " << distance.getCallCounter() << std::endl;
}

void AlgorithmKNN::constructGraph_reverseKNN(int repeat) {
    graph.resize(repeat);
    for (int r = 0; r < repeat; ++r) {
        tqdm bar;
        Points allPoints = points;
        points.clear();
        for (std::size_t i = 0; i < allPoints.size(); ++i) {
            std::vector<int> knn = findKNearestNeighborsMultiStart(allPoints[i], 2 * K, 3, false, r);
            int counter = 0;
            std::set<int> newEdges;
            for (std::size_t j = 0; j < std::min(size_t(K), knn.size()); ++j) {
                newEdges.insert(knn[j]);
            }
            for (auto j : newEdges) {
                graph[r][allPoints[i].id].push_back(Edge(allPoints[j].id));
                graph[r][allPoints[j].id].push_back(Edge(allPoints[i].id));
            }
            newEdges.clear();
            for (std::size_t j = std::min(size_t(K), knn.size()); j < knn.size(); ++j) {
                if (isPointTheNeighbor(allPoints[i], allPoints[knn[j]], K, r)) {
                    newEdges.insert(knn[j]);
                    ++counter;
                    //break;
                }
                if (counter == K)
                    break;
            }

            for (auto j : newEdges) {
                graph[r][allPoints[i].id].push_back(Edge(allPoints[j].id));
                graph[r][allPoints[j].id].push_back(Edge(allPoints[i].id));
            }
            points.push_back(allPoints[i]);
            bar.progress(int(i), int(allPoints.size()));
        }
    }
    std::cout << std::endl << "distance func call count: " << distance.getCallCounter() << std::endl;
}

bool AlgorithmKNN::isPointTheNeighbor(const Point& newPoint, const Point& oldPoint, int k, int r) {
    if (points.empty())
        return {};
    if (k == -1)
        k = K;
    std::size_t indexStartPoint = oldPoint.id;

    graph[r][indexStartPoint].push_back(Edge(newPoint.id));

    uint32_t dist = distance.calculateEuclideanDistance(points[indexStartPoint], newPoint);

    std::priority_queue<std::pair<uint32_t, int>, std::vector<std::pair<uint32_t, int>>, std::greater<std::pair<uint32_t, int>>> candidates;
    candidates.push({dist, indexStartPoint});

    std::unordered_map<std::size_t,int> visitedPoints;
    visitedPoints[indexStartPoint] = 1;

    std::priority_queue<std::pair<uint32_t, std::size_t>> topKNearestPoints;

    while(!candidates.empty()) {
        std::pair<uint32_t, std::size_t> currentPoint = candidates.top();
        candidates.pop();

        topKNearestPoints.push(currentPoint);
        if(int(topKNearestPoints.size()) == k + 1) {
            if (topKNearestPoints.top().second == currentPoint.second) {
                topKNearestPoints.pop();
                break;
            }
            topKNearestPoints.pop();
        }
        for(const Edge& edge : graph[r][currentPoint.second]) {
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
    graph[r][indexStartPoint].pop_back();
    return result;
}


std::vector<int> AlgorithmKNN::findKNearestNeighborsMultiStart(const Point& newPoint, int k, int repeat, bool age, int r) {
    if(k == -1)
        k = K;
    std::unordered_set<int> nearest_points;
    for(int i = 0; i < repeat; ++i) {
        std::vector<int> x = findKNearestNeighbors_Age(newPoint, k, age, r);
        for(int point : x) {
            nearest_points.insert(point);
        }
    }
    std::vector<std::pair<uint32_t,int>> dists;
    dists.reserve(nearest_points.size());
    for (int x : nearest_points) {
        dists.emplace_back(distance.calculateEuclideanDistance(points[x], newPoint), x);
    }
    std::sort(dists.begin(), dists.end());
    std::vector<int> result(std::min(size_t(k), dists.size()));
    for(std::size_t i = 0; i < std::min(size_t(k), dists.size()); ++i) {
        result[i] = dists[i].second;
    }
    return result;

}

std::vector<int> AlgorithmKNN::findKNearestNeighbors_Naive(const Point& newPoint) {
    std::priority_queue<std::pair<uint32_t, int>, std::vector<std::pair<uint32_t, int>>,
            std::greater<std::pair<uint32_t, int>>> topKNearestPoints;

    for(std::size_t i = 0; i < points.size(); ++i) {
        topKNearestPoints.push({distance.calculateEuclideanDistance(newPoint, points[i]), i});
    }

    std::vector<int> result(K);
    for(std::size_t i = 0; i < std::min(size_t(K),topKNearestPoints.size()); i++){
        result[i] = topKNearestPoints.top().second;
        topKNearestPoints.pop();
    }
    return result;
}

void AlgorithmKNN::addPoint(const Point& point) {
    std::vector<int> knn = findKNearestNeighbors(point);
    for(int x : knn) {
        graph[0][point.id].push_back(Edge(points[x].id));
        graph[0][points[x].id].push_back(Edge(point.id));
    }
    points.push_back(point);
}

uint64_t AlgorithmKNN::getCallDistanceCounter() const {
    return distance.getCallCounter();
}

std::vector<int> AlgorithmKNN::findKNearestNeighbors_Age(const Point& newPoint, int k, bool age, int r) {
    if(k == -1)
        k = K;
    return age ? findKNearestNeighbors(findOneNearestNeighborUsingAge(newPoint), k, r) :
           findKNearestNeighbors(newPoint, k, r);
}


Point AlgorithmKNN::findOneNearestNeighborUsingAge(const Point& newPoint, int r) {
    if (points.empty())
        return {};
    std::size_t indexCurPoint = distrib(gen) % points.size();
    uint32_t curAge = 1;
    uint32_t curDistance = distance.calculateEuclideanDistance(newPoint, points[indexCurPoint]);
    while(true) {
        bool newPointFounded = false;
        std::vector<Edge> goodEdges;
        for(const Edge& edge : graph[r][indexCurPoint]) {
            if(edge.age >= curAge) {
                goodEdges.push_back(edge);
            }
        }
        //std::sort(goodEdges.begin(), goodEdges.end(), [](const Edge& a, const Edge& b) { return a.age < b.age; });
        for(const Edge& edge : goodEdges) {
            uint32_t dist = distance.calculateEuclideanDistance(newPoint, points[edge.dest]);
            if(dist < curDistance) {
                indexCurPoint = edge.dest;
                curDistance = dist;
                curAge = edge.age;
                newPointFounded = true;
                //break;
            }
        }
        if(!newPointFounded) {
            break;
        }
    }
    return points[indexCurPoint];
}

std::vector<int> AlgorithmKNN::findKNearestNeighbors(const Point& newPoint, int k, int r) {
    if (points.empty())
        return {};
    if (k == -1)
        k = K;
    //std::vector<int> dist_delta;
    //std::vector<int> age_delta;
    std::size_t indexStartPoint = distrib(gen) % points.size();
    /*if(!graph[r][newPoint.id].empty())
        indexStartPoint = graph[r][newPoint.id].front().dest;*/
    uint32_t dist = distance.calculateEuclideanDistance(points[indexStartPoint], newPoint);
    //dist_delta.push_back(dist);
    std::priority_queue<std::pair<uint32_t, int>, std::vector<std::pair<uint32_t, int>>, std::greater<std::pair<uint32_t, int>>> candidates;
    candidates.push({dist, indexStartPoint});

    std::unordered_map<std::size_t,int> visitedPoints;
    visitedPoints[indexStartPoint] = 1;

    std::priority_queue<std::pair<uint32_t, int>> topKNearestPoints;

    while(!candidates.empty()) {
        std::pair<uint32_t, int> currentPoint = candidates.top();
        candidates.pop();

        topKNearestPoints.push(currentPoint);
        if(int(topKNearestPoints.size()) == k + 1) {
            if (topKNearestPoints.top().second == currentPoint.second) {
                topKNearestPoints.pop();
                break;
            }
            topKNearestPoints.pop();
        }
        for(const Edge& edge : graph[r][currentPoint.second]) {
            if(!visitedPoints[edge.dest]) {
                visitedPoints[edge.dest] = 1;
                dist = distance.calculateEuclideanDistance(points[edge.dest], newPoint);
                //dist_delta.emplace_back(dist);
                //age_delta.emplace_back(edge.age);
                candidates.push({dist, edge.dest});
            }
        }
    }
    std::vector<int> result;
    result.reserve(k);
    while(!topKNearestPoints.empty()) {
        result.push_back(topKNearestPoints.top().second);
        topKNearestPoints.pop();
    }/*
    std::ofstream output_file;
    output_file.open("age_delta.txt", std::fstream::app);
    for(auto& x : age_delta) {
        output_file << x << " ";
    }
    output_file << "\n";
    output_file.close();*/
    //std::reverse(result.begin(), result.end());
    return result;
}

std::vector<int> AlgorithmKNN::findKNearestNeighbors_multiGraph(const Point &newPoint, int k) {
    if(k == -1)
        k = K;
    std::unordered_map<int, int> counter;
    for (std::size_t r = 0; r < graph.size(); r++) {
        std::vector<int> x = findKNearestNeighborsMultiStart(newPoint, k, 3,false, r);
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
    for(std::size_t i = 0; i < std::min(size_t(k), t.size()); ++i) {
        result[i] = t[i].second;
    }
    return result;
}
