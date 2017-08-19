#include "loss.hpp"
#include <iostream>

namespace loss{
autodiff::var mean(autodiff::var& input, autodiff::var& target){
    auto loss = (input - target).asum();
    auto l_size = autodiff::var(nn::Tensor(loss.data.shape, input.size()));
    loss /= l_size;
    return loss;
}
autodiff::var mse(autodiff::var& input, autodiff::var& target){
    auto loss = (pow(input - target, 2)).asum();
    auto l_size = autodiff::var(nn::Tensor(loss.data.shape, input.size()));
    loss /= l_size;
    return loss;
}
autodiff::var crossEntropy(autodiff::var& input, autodiff::var& target){
    auto loss = -1 * (input * log(target)).sum();
    return loss;
}
} // namespace loss
