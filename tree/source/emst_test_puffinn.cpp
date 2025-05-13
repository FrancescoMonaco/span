#include "catch.hpp"
#include "puffinn/emst.hpp"
#include "puffinn/collection.hpp"
#include "puffinn/format/generic.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "H5Cpp.h"
#include <iostream>
#include <chrono>
#include <fstream>

using namespace puffinn;
const int MB = 1024*1024;

int main(){
    // const unsigned int DIMENSIONS = 8;
    // const unsigned int SIZE = 10000;

    // std::vector<std::vector<float>> data;
    // for (unsigned int i=0; i < SIZE; i++) {
    //     data.emplace_back(UnitVectorFormat::generate_random(DIMENSIONS));
    // }

    // Open a hdf5
    const std::vector<std::string> FILE_NAMES = {"/home/monaco/span/puffinn/glove-100-angular.hdf5",
                                                 "/home/monaco/span/puffinn/nytimes-256-angular.hdf5",
                                                 "/home/monaco/span/puffinn/lastfm-64-dot.hdf5"};
    const std::string FILE_NAME(FILE_NAMES[0]);
    const std::string DATASET_NAME("train");
    std::vector<std::vector<float>> data;
    try {
        // Open the HDF5 file
        H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);

        // Open the dataset
        H5::DataSet dataset = file.openDataSet(DATASET_NAME);

        // Get dataset dimensions
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[2]; // Assuming 2D dataset (rows, cols)
        dataspace.getSimpleExtentDims(dims, nullptr);
        size_t num_rows = dims[0];
        size_t num_cols = dims[1];

        // Allocate memory for reading
        data.resize(num_rows, std::vector<float>(num_cols));
        // Read the dataset into a 1D buffer
        std::vector<float> buffer(num_rows * num_cols);
        dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);

        // Reshape 1D buffer to 2D vector
        for (size_t i = 0; i < num_rows; ++i) {
            for (size_t j = 0; j < num_cols; ++j) {
                data[i][j] = buffer[i * num_cols + j];
                // if (i == 0) {
                //     std::cout << data[i][j] << " ";
                // }
            }
        }

        std::cout << "Successfully read " << num_rows << " rows and " << num_cols << " columns." << std::endl;
    } 
    catch (...) {
        std::cerr << "HDF5 Error: " << std::endl;
     }
      int DIMENSIONS = data[0].size();
    
    //Convert DIMENSIONS to a int32_t 
    uint32_t DIM = static_cast<uint32_t>(DIMENSIONS);
    std::cout << "DIMENSIONS: " << DIMENSIONS << " Num points: "<< data.size() << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    //  //for (size_t i = 0; i<3;i++ ){ 
         EMST emst(DIM, 900*MB, data);
         emst.find_epsilon_tree();   
     //}
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();///3;

    // True weight
     EMST emst_true(DIM, 8*MB, data);
     float tree_weight = emst_true.exact_tree();
     std::cout << "True tree weight: " << tree_weight << std::endl; 

     std::cout << "Time approx: " << duration << " mseconds" << std::endl;
}