#include "catch.hpp"
#include "puffinn/emst.hpp"
#include "puffinn/collection.hpp"

#include <iostream>

using namespace puffinn;

std::set<std::tuple<float, std::pair<unsigned int, unsigned int>>> top;
size_t num_data = 6; 
int MB = 1024*1024;

int main(){
    EMST emst(2, 1*MB, std::make_unique<std::vector<std::vector<float>>>(
        std::vector<std::vector<float>>{
            std::vector<float>{-1, 0},
            std::vector<float>{-1, -1},
            std::vector<float>{1, 0.15},
            std::vector<float>{1, 0.2},
            std::vector<float>{1, -0.1},
            std::vector<float>{4, -3},
            std::vector<float>{3.2, -3},
            std::vector<float>{4, 2.1},
            std::vector<float>{9.2, 72.3},
        }
    ));
    emst.find_epsilon_tree();

}