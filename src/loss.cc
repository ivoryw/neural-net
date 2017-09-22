#include "loss.hpp"
#include <iostream>

namespace loss{
autodiff::Var mean_error(autodiff::Var& input, autodiff::Var& target){
    auto loss = (input - target).abs_sum();
    auto l_size = autodiff::Var(nn::Tensor(loss.data.shape, input.size()));
    loss /= l_size;
    return loss;
}
autodiff::Var mean_squared_error(autodiff::Var& input, autodiff::Var& target){
    auto loss = (pow(input - target, 2)).abs_sum();
    auto l_size = autodiff::Var(nn::Tensor(loss.data.shape, input.size()));
    loss /= l_size;
    return loss;
}
autodiff::Var cross_entropy_loss(autodiff::Var& input, autodiff::Var& target){
    auto loss = -1 * (input * log(target)).sum();
    return loss;
}
} // namespace loss
