#include "tensor.hpp"

#include <Accelerate/Accelerate.h>

namespace nn{
double Tensor::asum(){ return cblas_dasum(size, data.get(), 1); }
}

