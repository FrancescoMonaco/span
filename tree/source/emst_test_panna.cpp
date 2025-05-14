#include <iostream>
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/distance.hpp"

using namespace panna;

int main()  {

    const size_t conc = 24;
    const size_t dimensions = 3;
    const size_t rep = 10;
    const size_t n = 1500;
    using Hasher = E2LSH<conc, NormedPoints>;

    E2LSHBuilder<conc, NormedPoints> builder ( dimensions );

    std::vector<std::vector<float>> points;

    for ( size_t i = 0; i < n; i++ ) {
        std::vector<float> point = sample_random_normal_vector( dimensions );
        points.push_back( point );
    }

    EMST<NormedPoints, Hasher, EuclideanDistance> tree( dimensions, rep, builder, points, dimensions );

    // Exact computation
     float weight_exact = tree.exact_tree();
     std::cout << "Exact weight is: " << weight_exact << std::endl;

    // Exact with predictions (?)
    (void) tree.find_tree();


    return 0;
}