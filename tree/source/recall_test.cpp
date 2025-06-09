#include <iostream>
#include <highfive/H5Easy.hpp>
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/distance.hpp"
#include "panna/data.hpp"
#include "panna/rand.hpp"
#include <chrono>
#include <string>

using namespace panna;

int main()  {

    std::string datasets[4] = {
        "fashion-mnist-784-euclidean.hdf5",
        "glove-100-angular.hdf5",
        "nytimes-256-angular.hdf5",
        "gist-960-euclidean.hdf5"
    };
    int index = 0;
    for (const auto& name: datasets) {
        std::cout << "Processing dataset: " << name << std::endl;

        seed_global_rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        // Parameters
        const size_t conc = 8;
        const uint8_t rotations = 3;
        const size_t dimensions[4] = { 784, 100, 256, 960 };
        size_t reps[2] = { 200, 500 };
        std::vector<float> weigths; 
        using Point = NormedPoints; // UnitNormPoints or NormedPoints
        using Distance = EuclideanDistance; // EuclideanDistance or AngularDistance or CosineDistance
        using Hasher = E2LSH<conc, Point>;
        //using Hasher = CrossPolytope<conc, Point, Distance, rotations>;

        for (const auto& rep: reps) {
            for (size_t i = 0; i < 3; i++) {
                E2LSHBuilder<conc, NormedPoints> builder(dimensions[index]);
                // CrossPolytopeBuilder<conc, Point, Distance, rotations> builder( dimensions[index] );

                H5Easy::File file(name, H5Easy::File::ReadOnly);
                std::vector<std::vector<float>> points =
                    H5Easy::load<std::vector<std::vector<float>>>(file, "/train");

                EMST<Point, Hasher, Distance> tree(dimensions[index], rep, builder, points);

                float weight = tree.find_tree();
                weigths.push_back(weight);
            }
            float recall = 0.0;
            auto min = *std::min_element(weigths.begin(), weigths.end());
            for (const auto& w: weigths) {
                if (w <= min) {
                    recall += 1.0;
                }
            }
            recall /= weigths.size();
            weigths.clear();
            // Open a file to write the results
            std::ofstream outfile("recall_results.txt", std::ios_base::app);
            outfile << name << ", " << rep
                    << ", " << recall << ", " << weigths[0] << ", " << weigths[1] << ", " << weigths[2] << std::endl;

        }
        index++;
    }
}