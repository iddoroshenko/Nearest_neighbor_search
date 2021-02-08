//
// Created by ilya on 04.02.2021.
//

#include "engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class Wrapper : public AlgorithmKNN {
public:
    Wrapper(int newK = 5) : AlgorithmKNN(newK) {};

    void pySetPoints(py::object input);

    py::array_t<int> pyFindKNearestNeighbors(py::object input, int k);

};


void Wrapper::pySetPoints(py::object input) {
    py::array_t < int32_t , py::array::c_style | py::array::forcecast > items(input);
    auto buffer = items.request();
    points.resize(buffer.shape[0]);
    int dim = buffer.shape[1];
    int id = 0;
    for (int i = 0; i < points.size(); ++i) {
        points[i].coordinates.resize(dim);
        const int* vector_data = items.data(i);
        for(int j = 0; j < dim; ++j)
            points[i][j] = vector_data[j];
        points[i].id = id++;
    }

}

py::array_t<int> Wrapper::pyFindKNearestNeighbors(py::object input, int k) {
    py::array_t < int32_t , py::array::c_style | py::array::forcecast > items(input);
    auto buffer = items.request();
    int dim = buffer.shape[1];
    Point newPoint;
    const int* vector_data = items.data(0);
    for(int j = 0; j < dim; ++j)
        newPoint.coordinates.push_back( vector_data[j] );
    newPoint.id = 0;
    auto result = py::array_t<int>(5);

    py::buffer_info buf = result.request();
    int *ptr = static_cast<int *>(buf.ptr);
    auto v = findKNearestNeighbors(newPoint, k);
    for (int i = 0; i < 5; i++)
        ptr[i] = v[i];
    return result;
}

PYBIND11_MODULE(engineWrapper, m) {

py::class_<Wrapper>(m, "Wrapper")
.def(py::init<int>())
.def("constructGraph", &Wrapper::constructGraph)
.def("constructGraph_reverseKNN", &Wrapper::constructGraph_reverseKNN)
.def("pySetPoints", &Wrapper::pySetPoints)
.def("pyFindKNearestNeighbors", &Wrapper::pyFindKNearestNeighbors);
}
