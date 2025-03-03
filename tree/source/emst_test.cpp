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

size_t num_data = 6; 
int MB = 1024*1024;

int main(){
    // std::vector<std::vector<float>> data = {       
    // std::vector<float>{-1, 0},      //0
    // std::vector<float>{-1, -1},     //1
    // std::vector<float>{1, 0.15},    //2
    // std::vector<float>{1, 0.2},     //3
    // std::vector<float>{1, -0.1},    //4
    // std::vector<float>{4, -3},      //5
    // std::vector<float>{3.2, -3},    //6
    // std::vector<float>{4, 2.1},     //7
    // std::vector<float>{9.2, 72.3},  //8
    // std::vector<float>{0.5 , -2.4}, //9
    // };

    // //Points have to be normalized
    // for (auto &point : data){
    //     float norm = 0;
    //     for (auto &coord : point){
    //         norm += coord*coord;
    //     }
    //     norm = sqrt(norm);
    //     for (auto &coord : point){
    //         coord /= norm;
    //     }
    // }

    const unsigned int DIMENSIONS = 3;
    const unsigned int SIZE = 100000;

    std::vector<std::vector<float>> data;
    for (unsigned int i=0; i < SIZE; i++) {
        data.emplace_back(UnitVectorFormat::generate_random(DIMENSIONS));
    }

    //Open glove-100-angular.hdf5
    // const std::string FILE_NAME("/home/monaco/span/puffinn/glove-100-angular.hdf5");
    // const std::string DATASET_NAME("train");
    // std::vector<std::vector<float>> data;
    // try {
    //     // Open the HDF5 file
    //     H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);

    //     // Open the dataset
    //     H5::DataSet dataset = file.openDataSet(DATASET_NAME);

    //     // Get dataset dimensions
    //     H5::DataSpace dataspace = dataset.getSpace();
    //     hsize_t dims[2]; // Assuming 2D dataset (rows, cols)
    //     dataspace.getSimpleExtentDims(dims, nullptr);
    //     size_t num_rows = dims[0];
    //     size_t num_cols = dims[1];

    //     // Allocate memory for reading
    //     data.resize(num_rows, std::vector<float>(num_cols));
    //     // Read the dataset into a 1D buffer
    //     std::vector<float> buffer(num_rows * num_cols);
    //     dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);

    //     // Reshape 1D buffer to 2D vector
    //     for (size_t i = 0; i < num_rows; ++i) {
    //         for (size_t j = 0; j < num_cols; ++j) {
    //             data[i][j] = buffer[i * num_cols + j];
    //         }
    //     }

    //     std::cout << "Successfully read " << num_rows << " rows and " << num_cols << " columns." << std::endl;
    // } 
    // catch (...) {
    //     std::cerr << "HDF5 Error: " << std::endl;
    // }


    //  int DIMENSIONS = data[0].size();
    //Convert DIMENSIONS to a int32_t use a cast
    uint32_t DIM = static_cast<uint32_t>(DIMENSIONS);
    std::cout << "DIMENSIONS: " << DIMENSIONS << " Num points: "<< data.size() << std::endl;

     auto t1 = std::chrono::high_resolution_clock::now();
     //for (size_t i = 0; i<3;i++ ){ 
         EMST emst(DIM, 400*MB, data);
         emst.find_epsilon_tree();   
     //}
     auto t2 = std::chrono::high_resolution_clock::now();
     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();///3;
     // True weight
    //  EMST emst_true(DIM, 8*MB, data);
    //  float tree_weight = emst_true.exact_tree();
    //  std::cout << "True tree weight: " << tree_weight << std::endl; 

     std::cout << "Time: " << duration << " mseconds" << std::endl;

}