#include "net.hpp"

namespace nn{
void Net::backward(const autodiff::Var& loss){
    loss.evaluate_leaves();
}
} // namespace nn
