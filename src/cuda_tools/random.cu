#include "random.cuh"

namespace ct 
{

template std::vector<int> random_uniform_selection<int>(std::vector<int> data, const uint n_sel, const uint seed);

}