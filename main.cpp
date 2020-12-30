#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <set>
#include <ctime>
#include <algorithm>

uint32_t maxAge = 1;

struct Edge{
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
        distance += uint64_t(point1[i] - point2[i]) * uint64_t(point1[i] - point2[i]);
    }
    return sqrt(distance);
}

adjacency_list constructGraph_Naive(const Points& points) {
    adjacency_list result(points.size());
    int id = 1;
    for(int i = 0; i < points.size(); ++i) {
        std::set<std::pair<int,int>> nearestK;
        for(int j = 0; j < points.size(); ++j) {
            if(i == j) continue;
            nearestK.insert({calculateDistance(points[i], points[j]), j});
            if(nearestK.size() == 6) {
                nearestK.erase(prev(nearestK.end()));
            }
        }
        for(auto& x : nearestK) {
            result[i].push_back(Edge(id, x.second));
            id++;
        }
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
        std::sort(goodEdges.begin(), goodEdges.end(), [](const Edge& a, const Edge& b) { return a.age > b.age; });
        for(const auto& edge : goodEdges) {
            long double dist = calculateDistance(points[indexCurPoint], points[edge.dest]);
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

int main() {
    srand(time(nullptr));
    std::ifstream file("../sample.txt");
    int n;
    file >> n;
    Points v(n);
    for(auto& x : v) {
        x.resize(2);
        file >> x[0] >> x[1];
    }
    auto graph = constructGraph_Naive(v);
    std::cout << addNewPoint(Point(1000, 395), v, graph);
    return 0;
}
