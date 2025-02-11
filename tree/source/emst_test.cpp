#include "catch.hpp"
#include "puffinn/emst.hpp"
#include "puffinn/collection.hpp"

#include <iostream>

using namespace puffinn;

std::set<std::tuple<float, std::pair<unsigned int, unsigned int>>> top;
size_t num_data = 6; 
int MB = 1024*1024;

int main(){
    std::vector<std::vector<float>> data = {       
    std::vector<float>{-1, 0},      //0
    std::vector<float>{-1, -1},     //1
    std::vector<float>{1, 0.15},    //2
    std::vector<float>{1, 0.2},     //3
    std::vector<float>{1, -0.1},    //4
    std::vector<float>{4, -3},      //5
    std::vector<float>{3.2, -3},    //6
    std::vector<float>{4, 2.1},     //7
    std::vector<float>{9.2, 72.3},  //8
    std::vector<float>{0.5 , -2.4}, //9
    };

    //Points have to be normalized
    for (auto &point : data){
        float norm = 0;
        for (auto &coord : point){
            norm += coord*coord;
        }
        norm = sqrt(norm);
        for (auto &coord : point){
            coord /= norm;
        }
    }

    EMST emst(2, 0.3*MB, data);
    emst.find_epsilon_tree();

}