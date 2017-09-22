#include "loss.hpp"
#include <iostream>

namespace loss{
    
using autodiff::Var;
using nn::Tensor;

Var l1_loss(Var& input, Var& target) {
    auto loss = (input - target).abs_sum();
    return loss;
}

Var mean_error(Var& input, Var& target) {
    auto loss = (input - target).abs_sum();
    auto l_size = Var(Tensor(loss.data.shape, input.size()));
    loss /= l_size;
    return loss;
}
Var mean_squared_error(Var& input, Var& target) {
    auto loss = (pow(input - target, 2)).abs_sum();
    auto l_size = Var(Tensor(loss.data.shape, input.size()));
    loss /= l_size;
    return loss;
}
Var cross_entropy_loss(Var& input, Var& target) {
    auto loss = -1 * (input * log(target)).sum();
    return loss;
}
} // namespace loss
