/**
 * @file weighted_polytope.hpp
 * @brief Contains methods to compute various properties of a polytope with weighted points.
 */

#ifndef POLYTOPE_HPP_
#define POLYTOPE_HPP_

#include <vector>
using std::vector;

#include <CGAL/Epick_d.h>
#include <CGAL/Regular_triangulation.h>


namespace PolytopeCPP {


// Define dynamic Kernel
using DDim              = CGAL::Dynamic_dimension_tag;
using K                 = CGAL::Epick_d<DDim>;

using Triangulation     = CGAL::Regular_triangulation<K>;

using Point             = K::Point_d;
using WeightedPoint     = K::Weighted_point_d;
using Facet             = Triangulation::Facet;

using Real              = float;


enum PrintingFormat {
    NONE    = 0,
    PYTHON  = 1
};


class Polytope {
 private:
    Triangulation _t;
    int _dim;
    PrintingFormat _format;


 public:
    Polytope(int dim, vector<WeightedPoint>& points) : _t(dim), _dim(dim), _format(NONE) {
        _t.insert(points.begin(), points.end());
    }


    Polytope(int dim, vector<Point>& points) : _t(dim), _dim(dim), _format(NONE) {
        vector<WeightedPoint> w_points;
        for (auto p : points)
            w_points.push_back(WeightedPoint(p, 1.0f));
        _t.insert(w_points.begin(), w_points.end());
    }


    Polytope(int dim, vector<Point>& points, vector<Real>& heights) : _t(dim), _dim(dim), _format(NONE) {
        if (points.size() != heights.size()) throw std::runtime_error(
            "Array size mismatch between \"points\" and \"heights\". Received: points.size() = " +
                std::to_string(points.size()) + " and heights.size() = " + std::to_string(heights.size()));


        for (int j = 0; j < points.size(); ++j) {
            Real h0 = 0;
            for (int k = 0; k < _dim; ++k) {
                h0 += points.at(j)[k]*points.at(j)[k];
            }
            heights[j] = h0 - heights[j];
        }


        vector<WeightedPoint> w_points;
        for (int j = 0; j < points.size(); ++j)
            w_points.push_back(WeightedPoint(points.at(j), heights.at(j)));

        _t.insert(w_points.begin(), w_points.end());
    }


    /**
     * @brief Get the finite points object
     * @return Sign of the orientation.
     */
    int get_finite_points(Facet& facet, vector<WeightedPoint>& out) {
        int count = 0;
        // Compute boundary intersected w/ complement of infinite_vertex.
        int sign = 1.0;
        for (auto v_iter = facet.full_cell()->vertices_begin(); v_iter != facet.full_cell()->vertices_end(); ++v_iter) {
            if (*v_iter == _t.infinite_vertex()) {
                sign = pow(-1, count);
                continue;
            }
            out.push_back((*v_iter)->point());
            ++count;
        }

        return sign;
    }


    void get_facets(vector<Facet>& facets, bool only_adjacent_to_inf = true) {
        for (auto cell_iter = _t.full_cells_begin(); cell_iter != _t.full_cells_end(); ++cell_iter) {
            // Only pick facets adjacent to the infinite vertex
            if (!_t.is_infinite(cell_iter) && only_adjacent_to_inf) continue;
            facets.push_back(
                Facet(cell_iter, cell_iter->index(_t.infinite_vertex())));
        }
    }


    void get_facets(vector< vector<Point> >& facets, bool only_adjacent_to_inf = true) {
        for (auto cell_iter = _t.full_cells_begin(); cell_iter != _t.full_cells_end(); ++cell_iter) {
            if (!_t.is_infinite(cell_iter) && only_adjacent_to_inf) continue;
            vector<Point> points;
            // for (auto v_iter = cell_iter->vertices_begin(); v_iter != cell_iter->vertices_end(); ++v_iter) {
            //     if ((*v_iter)->point().point().size() > 0)
            //         points.push_back((*v_iter)->point().point());
            // }

            for (int j = 0; j < _dim + 1; ++j) {
                points.push_back(cell_iter->vertex(j)->point().point());
            }
            facets.push_back(points);
        }
    }


    double determinant(Facet& facet) {
        vector<WeightedPoint> finite_points;
        int sign = get_finite_points(facet, finite_points);

        Eigen::MatrixXf vol_mat(_t.current_dimension(), _t.current_dimension());
        for (int i = 0; i < finite_points.size(); ++i) {
            auto p = finite_points.at(i).point();
            vector<Real> coords(
                p.cartesian_begin(),
                p.cartesian_end());
            vol_mat.row(i) = Eigen::Map<Eigen::VectorXf>(coords.data(), coords.size());
        }

        return sign*(vol_mat.determinant());
    }


    /**
     * @brief Computes volume of the convex hull.
     * @note THe normalization coefficient 1/dim! is omitted.
     * @return Signed volume
     */
    double compute_volume() {
        double volume = 0.0f;
        const int dim = _t.current_dimension();

        for (auto cell_iter = _t.full_cells_begin(); cell_iter != _t.full_cells_end(); ++cell_iter) {
            // Only pick facets adjacent to the infinite vertex
            if (!_t.is_infinite(cell_iter)) continue;
            Facet facet(cell_iter, cell_iter->index(_t.infinite_vertex()));
            volume += determinant(facet);
        }

        return volume; /// tgamma(dim + 1);
    }


    Polytope& operator<<(PrintingFormat format) {
        this->_format = format;
        return *this;
    }


    friend std::ostream& operator<<(std::ostream& stream, Polytope& other) {
        bool pyformat = (other._format == PrintingFormat::PYTHON);
        if (pyformat) stream << "[";
        for (auto cell_iter = other._t.full_cells_begin(); cell_iter != other._t.full_cells_end(); ++cell_iter) {
            // Only pick facets adjacent to the infinite vertex
            if (!other._t.is_infinite(cell_iter)) continue;
            Facet facet(cell_iter, cell_iter->index(other._t.infinite_vertex()));

            if (pyformat) stream << "[";
            for (auto v_iter = facet.full_cell()->vertices_begin(); v_iter != facet.full_cell()->vertices_end(); ++v_iter) {
                // Skip infinite vertex
                if (*v_iter == other._t.infinite_vertex()) continue;
                if (pyformat) stream << "[";
                for (int k = 0; k < other._dim; ++k) {
                    stream << (*v_iter)->point().point()[k] << ((k == other._dim - 1) ? "" : ",");
                }
                if (pyformat) stream << ((std::next(v_iter) == facet.full_cell()->vertices_end()) ? "]" : "],");
                else stream << "\n";
            }
            if (pyformat) stream << ((std::next(cell_iter) == other._t.full_cells_end()) ? "]" : "],\n");
            else stream << "\n";
        }
        if (pyformat) stream << "]\n";

        return stream;
    }

};



}  // PolytopeCPP



#endif  // POLYTOPE_HPP_
