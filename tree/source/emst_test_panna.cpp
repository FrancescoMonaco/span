#include <iostream>
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/distance.hpp"

using namespace panna;

int main()  {

    using Hasher = E2LSH<24, NormedPoints>;
    const size_t dimensions = 100;
    const size_t rep = 10;
    const size_t n = 100;

    E2LSHBuilder<24, NormedPoints> builder ( dimensions );

    std::vector<std::vector<float>> points;

    for ( size_t i = 0; i < n; i++ ) {
        std::vector<float> point = sample_random_normal_vector( dimensions );
        points.push_back( point );
    }

    EMST<NormedPoints, Hasher, EuclideanDistance> tree( dimensions, rep, builder, points, dimensions );

    float weight_exact = tree.exact_tree();

    (void) tree.find_tree();

    std::cout << "Exact weight is: " << weight_exact << std::endl;

    return 0;
}