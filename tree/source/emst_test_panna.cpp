#include <iostream>
#include <highfive/H5Easy.hpp>
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/distance.hpp"
#include "panna/data.hpp"
#include "panna/rand.hpp"
#include <chrono>

using namespace panna;

int main()  {

    const size_t conc = 8;
    const size_t dimensions = 256;
    const size_t rep = 200;
    const size_t n = 100000;
    seed_global_rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );
    using Hasher = E2LSH<conc, NormedPoints>;
    //using Hasher = CrossPolytope<conc, UnitNormPoints, CosineDistance>;

    E2LSHBuilder<conc, NormedPoints> builder ( dimensions );
    //CrossPolytopeBuilder<conc, UnitNormPoints, CosineDistance> builder( dimensions );

    // std::vector<std::vector<float>> points;
    // for ( size_t i = 0; i < n; i++ ) {
    //     std::vector<float> point = sample_random_normal_vector( dimensions );
    //     points.push_back( point );
    // }
    // H5Easy::File file( "fashion-mnist-784-euclidean.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "glove-100-angular.hdf5", H5Easy::File::ReadOnly );
    H5Easy::File file( "nytimes-256-angular.hdf5", H5Easy::File::ReadOnly );

    std::vector<std::vector<float>> points =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );
    //points.resize( n );


    EMST<NormedPoints, Hasher, EuclideanDistance> tree( dimensions, rep, builder, points );

    // Exact computation
    // auto start_exact = std::chrono::high_resolution_clock::now();
    //  float weight_exact = tree.exact_tree();
    // auto end_exact = std::chrono::high_resolution_clock::now();
    //  std::cout << "Exact weight is: " << weight_exact << " in " << std::chrono::duration<double>(end_exact - start_exact).count() << " seconds" << std::endl;

    // Exact with predictions (?)
    auto start = std::chrono::high_resolution_clock::now();
    // for (size_t iter= 0; iter< 3 ; iter++) {
    //     EMST<NormedPoints, Hasher, EuclideanDistance> tree( dimensions, rep, builder, points );

        (void) tree.find_tree();
    // }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = (end - start);
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    // std::cout << "----- Epsilon Version -----" << std::endl;
    // // Approximate with predictions
    // (void) tree.find_epsilon_tree();


    return 0;
}