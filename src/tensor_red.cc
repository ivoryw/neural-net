#include "tensor.hpp"

#include <Accelerate/Accelerate.h>

namespace nn{
double Tensor::asum(){ return cblas_dasum(size, data.get(), 1); }
double Tensor::sum(){
    auto C = std::make_unique<double>();
    vDSP_sveD(data.get(), 1, C.get(), size);
    return *C;
}
}

