#include <iostream>
#include <highfive/H5Easy.hpp>
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/distance.hpp"
#include "panna/data.hpp"
#include "panna/rand.hpp"

using namespace panna;

int main()  {

    const size_t conc = 8;
    const size_t dimensions = 784;
    const size_t rep = 200;
    const size_t n = 1500;
    using Hasher = E2LSH<conc, NormedPoints>;
    //using Hasher = CrossPolytope<conc, UnitNormPoints, CosineDistance>;

    E2LSHBuilder<conc, NormedPoints> builder ( dimensions );
    //CrossPolytopeBuilder<conc, UnitNormPoints, CosineDistance> builder( dimensions );

    // std::vector<std::vector<float>> points;
    // for ( size_t i = 0; i < n; i++ ) {
    //     std::vector<float> point = sample_random_normal_vector( dimensions );
    //     points.push_back( point );
    // }
    H5Easy::File file( "fashion-mnist-784-euclidean.hdf5", H5Easy::File::ReadOnly );

    std::vector<std::vector<float>> points =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );


    EMST<NormedPoints, Hasher, EuclideanDistance> tree( dimensions, rep, builder, points );

    // Exact computation
    //  float weight_exact = tree.exact_tree();
    //  std::cout << "Exact weight is: " << weight_exact << std::endl;

    // Exact with predictions (?)
    (void) tree.find_tree();

    // std::cout << "----- Epsilon Version -----" << std::endl;
    // // Approximate with predictions
    // (void) tree.find_epsilon_tree();


    return 0;
}