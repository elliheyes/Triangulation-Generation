#include <iostream>
using std::cout;

#include <memory>
using std::shared_ptr;

#include <vector>
using std::vector;

#include <weighted_polytope.hpp>
using PolytopeCPP::WeightedPoint;
using PolytopeCPP::Point;
using PolytopeCPP::Real;

#include <boost/python.hpp>

namespace py = boost::python;

template <typename T>
vector<T> to_vector(const py::object& iterable) {
    return vector<T>(
        py::stl_input_iterator<T>(iterable),
        py::stl_input_iterator<T>());
}


class Polytope {
 private:
    typedef shared_ptr<PolytopeCPP::Polytope> ptr_Polytope;
    ptr_Polytope polytope;

 public:
    Polytope(const py::object& vertices, const py::object& heights = py::object()) {
        vector<WeightedPoint> points;

        int dim = -1;

        auto vertices_vect = to_vector<py::object>(vertices);
        vector<Real> heights_vect;
        if (heights != py::object()) {
            heights_vect = to_vector<Real>(heights);
        } else {
            heights_vect = std::vector<Real>(vertices_vect.size(), 1.0f);
        }

        if (vertices_vect.size() != heights_vect.size()) throw std::runtime_error(
            "Array size mismatch between \"points\" and \"heights\". Received: points.size() = " +
                std::to_string(vertices_vect.size()) + " and heights.size() = " + std::to_string(heights_vect.size()));


        for (int j = 0; j < vertices_vect.size(); ++j) {
            auto coords = vertices_vect.at(j);
            auto height = heights_vect.at(j);
            auto coords_vect =  to_vector<Real>(coords);
            if (dim == -1)
                dim = coords_vect.size();
            else if (dim != coords_vect.size())
                throw std::runtime_error("Inconsistent dimension. Expected: " + std::to_string(dim) + " but received: " + std::to_string(coords_vect.size()));

            points.push_back(
                WeightedPoint(
                    Point(dim, coords_vect.data(), coords_vect.data() + coords_vect.size()), height));
        }

        polytope = ptr_Polytope(new PolytopeCPP::Polytope(dim, points));
    }


    py::object get_facets() {
        py::list facets;

        vector<vector<Point>> facets_vect;
        this->polytope->get_facets(facets_vect);

        for (auto& pts : facets_vect) {
            py::list points;
            for (auto& pt : pts) {
                py::list coords;
                for (auto c_iter = pt.cartesian_begin(); c_iter != pt.cartesian_end(); ++c_iter) {
                    coords.append(*c_iter);
                }
                points.append(coords);
            }
            facets.append(points);
        }

        return facets;
    }


    py::tuple get_oriented_facets_of_hull() {
        py::list facets;
        py::list orientations;

        vector<PolytopeCPP::Facet> facets_vect;

        this->polytope->get_facets(facets_vect);

        for (PolytopeCPP::Facet& f : facets_vect) {
            vector<WeightedPoint> pts;
            int orientation = this->polytope->get_finite_points(f, pts);
            orientations.append(orientation);

            py::list points;
            for (auto& wp : pts) {
                py::list coords;
                auto p = wp.point();
                for (auto c_iter = p.cartesian_begin(); c_iter != p.cartesian_end(); ++c_iter) {
                    coords.append(*c_iter);
                }
                points.append(coords);
            }
            facets.append(points);
        }

        return py::make_tuple(facets, orientations);
    }
};


BOOST_PYTHON_MODULE(py_polytope) {
    py::class_<Polytope>("Polytope", py::init<py::object, py::object>())
        .def("getFacets",               &Polytope::get_facets);
}
