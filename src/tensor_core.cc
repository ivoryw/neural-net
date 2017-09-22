#include "tensor.hpp"

#include <numeric>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <atlas.h>
#endif // __APPLE__

namespace nn {

using std::make_unique;
using std::invalid_argument;

Tensor::Tensor(size_t x, size_t y, size_t z, size_t t)
    : size(x * y * z * t), shape{{x,y,z,t}} {
    if((x || y || z || t) == 0) throw invalid_argument("Tensor dimensions must be non-zero");
    data = make_unique<double[]>(size);
}

Tensor::Tensor(const Shape& shape_) : shape(shape_) {
    size = accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>());
    data = make_unique<double[]>(size);
}

Tensor::Tensor(const Shape& shape_, double constant) : shape(shape_) {
    size = accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>());
    data = make_unique<double[]>(size);
    this->constant(constant);
}
Tensor::Tensor(const Tensor& rhs) : size(rhs.size), shape(rhs.shape) {
    data = make_unique<double[]>(size);
    auto size_i = static_cast<int>(size);
    cblas_dcopy(size_i, rhs.data.get(), 1, data.get(), 1);
}
Tensor::Tensor()
    : size(1), shape{{1,1,1,1}} {
    data = make_unique<double[]>(1);
}

double &Tensor::operator()(size_t x, size_t y, size_t z, size_t t) {
    if(shape[0] <= x) throw invalid_argument("x is outside the tensor");
    if(shape[1] <= y) throw invalid_argument("y is outside the tensor");
    if(shape[2] <= z) throw invalid_argument("z is outside the tensor");
    if(shape[3] <= t) throw invalid_argument("t is outside the tensor");
    auto z_mult = shape[0] + shape[1];
    auto t_mult = z_mult + shape[2];
    return data[x + y*shape[0] + z * (z_mult) + t * (t_mult)];
}
double Tensor::operator()(size_t x, size_t y, size_t z, size_t t)const {
    if(shape[0] <= x) throw invalid_argument("x is outside the tensor");
    if(shape[1] <= y) throw invalid_argument("y is outside the tensor");
    if(shape[2] <= z) throw invalid_argument("z is outside the tensor");
    if(shape[3] <= t) throw invalid_argument("t is outside the tensor");
    auto z_mult = shape[0] + shape[1];
    auto t_mult = z_mult + shape[2];
    return data[x + y*shape[0] + z * (z_mult) + t * (t_mult)];
}

nn::Tensor Tensor::row(size_t y) {
    nn::Tensor r(shape[0]);
    for(size_t i = 0; i < shape[0]; ++i) {
        r(i) = data[i + shape[1] * y];
    }
    return r;
}

nn::Tensor Tensor::col(size_t x) {
    nn::Tensor c(1, shape[1]);
    for(size_t i = 0; i < shape[1]; ++i) {
        c(0, i) = data[shape[0] * x + i];
    }
    return c;
}

nn::Tensor Tensor::slice(size_t z) {
    nn::Tensor s(shape[0], shape[1]);
    for(size_t i = 0; i < shape[0]; ++i) {
        for(size_t j = 0; j < shape[1]; ++j) {
            s(i, j) = data[i + j * shape[0] + z * (shape[0] * shape[1])];
        }
    }
    return s;
}

} // namespace nn
