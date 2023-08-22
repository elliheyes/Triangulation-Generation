#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using std::cout;

#include <fstream>
using std::ofstream;
using std::ifstream;

#include <exception>

#include <vector>
using std::vector;

#include <string>
using std::string;

#include <sstream>
using std::istringstream;

#include <math.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;


#include <weighted_polytope.hpp>

#include <CGAL/Epick_d.h>
#include <CGAL/Regular_triangulation.h>

// Define dynamic Kernel
using DDim              = CGAL::Dynamic_dimension_tag;
using K                 = CGAL::Epick_d<DDim>;
using Triangulation     = CGAL::Regular_triangulation<K>;
using Point             = K::Point_d;
using WeightedPoint     = K::Weighted_point_d;
using Facet             = Triangulation::Facet;

typedef float Real;


void load_points(string path_to_vertices, vector<WeightedPoint>& out, int& dim) {
    ifstream f(path_to_vertices);
    if (!f) {
        throw std::runtime_error("Unable to open file: " + path_to_vertices);
    }

    dim = -1;

    string line;
    while (f >> line) {
        istringstream ss(line);
        string token;
        vector<Real> coords;
        while(std::getline(ss, token, ',')) {
            coords.push_back(atof(token.c_str()));
        }

        if (dim == -1)
            dim = coords.size();
        else if (dim != coords.size())
            throw std::runtime_error("Inconsistent dimension for line '" + line + "'. Expected: " + std::to_string(dim) + " but received: " + std::to_string(coords.size()));

        out.push_back(
            WeightedPoint(
                Point(coords.size(), coords.data(), coords.data() + coords.size()), (1.0*random())/RAND_MAX));
    }
}


int main(int argc, char** argv) {
    // Parse args & load vertices
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h",      "Show this help message.")
    ("input,i",     po::value<string>()->required(),          "Path to the file containing vertices.")
    ("pyformat,f",  "If specified, uses python array format for output.")
    ("output,o",    po::value<string>(),                      "Path to the output.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return EXIT_SUCCESS;
    }

    try {
        po::notify(vm);
    } catch (std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    vector<WeightedPoint> points;
    int dim;
    string path_to_vertices = vm["input"].as<string>();
    try {
        load_points(path_to_vertices, points, dim);
    } catch (std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }


    PolytopeCPP::Polytope p(dim, points);
    cout << p.compute_volume() << "\n";


    vector< vector<Point> > facets;
    p.get_facets(facets);

    cout << (p << PolytopeCPP::PrintingFormat::PYTHON) << "\n";
    return EXIT_SUCCESS;
}
