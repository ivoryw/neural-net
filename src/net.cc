#include "net.hpp"

namespace nn{
void Net::backwardProp(const autodiff::var& loss){
    loss.evaluateLeaves();
}

} // namespace nn
