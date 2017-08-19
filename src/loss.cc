#include "loss.hpp"
#include <iostream>

namespace loss{
autodiff::var mean(autodiff::var& input, autodiff::var& target){
    auto loss = (input - target).asum();
    auto l_size = autodiff::var(nn::Tensor(loss.data.shape, input.size()));
    loss /= l_size;
    return loss;
}
}
