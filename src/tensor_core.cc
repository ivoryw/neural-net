/**
    Tensor core
    Intitialisers and access operators for tensor objects
*/
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

Tensor::Tensor(size_t x, size_t y, size_t z, size_t t) : size(x * y * z * t), shape{{x,y,z,t}} {
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
    // 
    auto z_mult = shape[0] + shape[1];
    auto t_mult = z_mult + shape[2];
    return data[x + y*shape[0] + z * (z_mult) + t * (t_mult)];
}
double Tensor::operator()(size_t x, size_t y, size_t z, size_t t) const {
    if(shape[0] <= x) throw invalid_argument("x is outside the tensor");
    if(shape[1] <= y) throw invalid_argument("y is outside the tensor");
    if(shape[2] <= z) throw invalid_argument("z is outside the tensor");
    if(shape[3] <= t) throw invalid_argument("t is outside the tensor");
    auto z_mult = shape[0] + shape[1];
    auto t_mult = z_mult + shape[2];
    return data[x + y*shape[0] + z * (z_mult) + t * (t_mult)];
}

Tensor Tensor::row(size_t y) {
    if(y >= data[1]) throw invalid_argument("y is outside the matrix");
    Tensor r(shape[0]);
    auto data_start = data.get() + y * shape[0];
    cblas_dcopy(shape[0], data_start, 1, r.data.get(), 1);
    return r;
}

Tensor Tensor::col(size_t x) {
    if(x >= data[0]) throw invalid_argument("x is outside the matrix");
    Tensor c(1, shape[1]);
    cblas_dcopy(shape[1], data.get() + x, shape[0], c.data.get(), 1);
    return c;
}

Tensor Tensor::slice(size_t z) {
    if(z >= data[2]) throw invalid_argument("z is outside the tensor");
    Tensor s(shape[0], shape[1]);
    cblas_dcopy(shape[0] * shape[1], data.get() + s.size * z, 1, s.data.get(), 1);
    return s;
}

Tensor Tensor::tube(size_t x_0, size_t y_0, size_t x_1, size_t y_1) {
    if(((x_0 || x_1) >= shape[0]) || ((y_0 || y_1) >= shape[1])) {
        throw invalid_argument("Tube coordinates are outside the tensor");
    }
    auto x = x_1 - x_0 + 2;
    auto y = y_1 - y_0 + 2;
    Tensor t(x, y, shape[2]);
    for(size_t i = 0; i < shape[2]; ++i) {
        t.set_slice(i, this->slice(i).sub_mat(x_0, y_0, x_1, y_1));
    }
    return t;
}

Tensor Tensor::sub_mat(size_t x_0, size_t y_0, size_t x_1, size_t y_1) {
    if(((x_0 || x_1) >= shape[0]) || ((y_0 || y_1) >= shape[1])) {
        throw invalid_argument("Tube coordinates are outside the tensor");
    }
    auto x = x_1 - x_0 + 2;
    auto y = y_1 - y_0 + 2;
    Tensor s(x, y);
    for(size_t i = 0; i < y; ++i){
        auto row_ptr = data.get() + x + i * shape[0];
        auto s_row_ptr = s.data.get() + i * x;
        cblas_dcopy(x, row_ptr, 1, s_row_ptr, 1);
    } 
    return s;
}

} // namespace nn
