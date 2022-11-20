
#include "test_random.h"

#include <cuda_tools/random.h>

#include <iostream>
#include <numeric>
#include <vector>

#include "cuda_tools/timer.cuh"

template <class T>
std::vector<T> random_selection_cpu(const std::vector<T>& data, size_t num_random)
{
    std::vector<T> selected;
    selected.reserve(num_random);
    const size_t n = data.size();
    while (num_random--)
    {
        auto it = data.begin();
        std::advance(it, rand() % n);
        selected.push_back(*it);
    }
    return selected;
}

TEST_F(TestRandom, test_random_selection)
{
    // uint seed = 0;

    uint n = 100000;
    uint n_sel = 10000;
    uint seed = 0;

    std::vector<int> data(n);
    std::iota(data.begin(), data.end(),
              0);  // initialize data

    // std::vector<int> sel_data = ct::random_uniform_selection<int>(data, n_sel);

    std::vector<int> sel_data = ct::timer(ct::random_uniform_selection<int>, data, n_sel, seed);
    
    std::vector<int> sel_data_cpu = ct::timer<std::chrono::microseconds>(random_selection_cpu<int>, data, n_sel);

    std::cout << "Done!\n";
    // for (auto i : sel_data)
    // {
    //     std::cout << i << "\n";
    // }
}