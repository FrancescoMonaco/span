#include <omp.h>
#include <puffinn.hpp>
#include <catch.hpp>
#include <stdio.h>
#include <random>
#include <chrono>
#include <thread>

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, 10);
    auto num = dist(gen);

#pragma omp parallel
    #pragma omp critical
    {
        printf("Hello from thread %d\n", omp_get_thread_num());
    }

    #pragma omp master
        printf("Hello from master thread\n");

    return 0;

}
