#include "catch.hpp"
#include "puffinn/emst.hpp"
#include "puffinn/collection.hpp"
#include "puffinn/format/generic.hpp"
#include "puffinn/similarity_measure/cosine.hpp"

#include <iostream>
#include <chrono>
#include <fstream>

using namespace puffinn;

std::set<std::tuple<float, std::pair<unsigned int, unsigned int>>> top;
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

    // const unsigned int DIMENSIONS = 2;
    // const unsigned int SIZE = 100;

    // std::vector<std::vector<float>> data;
    // for (unsigned int i=0; i < SIZE; i++) {
    //     data.emplace_back(UnitVectorFormat::generate_random(DIMENSIONS));
    // }

    //Open glove-100-angular.hdf5
    std::vector<std::vector<float>> data;
    std::ifstream file("glove-100-angular.hdf5", std::ios::binary);
    if (file.is_open()){
        std::vector<float> point(100);
        Dataset<UnitVectorFormat> dataset(100);
        while (file.read(reinterpret_cast<char*>(point.data()), 100*sizeof(float))){
            float norm = 0;
                for (auto &coord : point){
                    norm += coord*coord;
                }
                norm = sqrt(norm);
                for (auto &coord : point){
                    if  (coord == 0){
                        continue;
                    }
                    coord /= norm;
                }
            // If any value is outside -1 and 1 print a warning
            for (auto val : point){
                if (val < -1 || val > 1){
                    std::cout << "Value out of range: " << val << std::endl;
                }
            }
            //auto norm_point = to_stored_type<UnitVectorFormat>(point, dataset.get_description());



            data.push_back(point);
        }
    }
    else{
        std::cout << "File not found" << std::endl;
    }

    int DIMENSIONS = data[0].size();
    //Convert DIMENSIONS to a int32_t use a cast
    uint32_t DIM = static_cast<uint32_t>(DIMENSIONS);
    std::cout << "DIMENSIONS: " << DIMENSIONS << " Num points: "<< data.size() << std::endl;

    // auto t1 = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i<3;i++ ){ 
         EMST emst(DIM, 70*MB, data);
         emst.find_epsilon_tree();    
    // }
    // auto t2 = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count()/3;
    // std::cout << "Time: " << duration << " mseconds" << std::endl;

}