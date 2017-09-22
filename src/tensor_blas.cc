#include "tensor.hpp"

#include <algorithm>
#include <numeric>
#include <iomanip>
#include <random>
#include <iostream>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <atlas.h>
#endif // __APPLE__

using std::make_unique;
using std::string;
using std::accumulate;
using std::invalid_argument;

namespace nn {
static std::random_device rd;
static std::mt19937 eng(rd());
static string size_err {"Tensor sizes do not match"};

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

void Tensor::operator=(const Tensor& rhs) {
    if(shape != rhs.shape) throw invalid_argument(size_err);
    auto size_i= static_cast<int>(size);
    cblas_dcopy(size_i, rhs.data.get(), 1, data.get(),1);
}

Tensor Tensor::operator+(const Tensor& rhs) const {
    if(shape != rhs.shape) throw invalid_argument(size_err);
    Tensor result(*this);
    auto size_i = static_cast<int>(size);
    catlas_daxpby(size_i, 1, rhs.data.get(), 1, 1, result.data.get(), 1);
    return result;
}

Tensor Tensor::operator-(const Tensor& rhs)const {
    if(shape != rhs.shape) throw invalid_argument(size_err);
    Tensor result(*this);
    auto size_i = static_cast<int>(size);
    catlas_daxpby(size_i, 1, rhs.data.get(), 1, -1, result.data.get(), 1);
    return result;
}

// Tensor multiplication is messier due to the decision tree for BLAS funcs so
// isn't inlined like the rest
Tensor Tensor::operator*(const Tensor& rhs) const {
    Tensor result(rhs.shape[0], shape[1]);
    auto A = data.get();
    auto A_c = static_cast<int>(shape[0]);
    auto A_r = static_cast<int>(shape[1]);
    auto X = rhs.data.get();
    auto X_c = static_cast<int>(rhs.shape[0]);
    auto X_r = static_cast<int>(rhs.shape[1]);
    auto Y = result.data.get();
    auto Y_c = static_cast<int>(result.shape[0]);
    auto Y_r = static_cast<int>(result.shape[1]);

    if( rhs.size == 1) {
        return rhs.data[0] * *this;
    }
    else if( size == 1) {
        return data[0] * rhs;
    }
    else if((rhs.shape[2] && rhs.shape[3]) == 1) {
        if(A_r != Y_r || X_c != Y_c || A_c != X_r) {
            throw invalid_argument(size_err);
        }
        if(rhs.shape[1] == 1) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, A_r, A_c, 1, A, A_c, X, 1, 1, Y, 1);
        }
        else {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_r, X_c, A_c, 1, A, A_c, X, X_c, 1, Y, Y_c);
        }
    }
    else if(rhs.shape[3] == 1) {
        for(size_t i = 0; i < shape[2]; ++i) {
            auto slice_ptr = A + (shape[2] * i);
            auto rhs_ptr = X + (shape[2] * i);
            auto result_ptr = Y+(shape[2]*i);
            cblas_dgemm(CblasRowMajor, CblasNoTrans,CblasNoTrans,
                        A_c, X_c, A_c,
                        1, slice_ptr, A_c,
                        rhs_ptr, X_c, 1,
                        result_ptr, Y_c);
        }
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& rhs) const {
    if(shape != rhs.shape) throw invalid_argument(size_err);
    Tensor result(shape);
#ifdef __APPLE__
    auto size_ = static_cast<int>(size);
    auto r_ptr = result.data.get();
    auto rhs_ptr = rhs.data.get();
    auto t_ptr = data.get();
    // vecLib has a nice element-wise reciprocal function
    vvrec(r_ptr, rhs_ptr, &size_);
    // Then just perform a diagonal matrix mult
    auto size_i = static_cast<int>(size);
    cblas_dtbmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, size_i, 0, t_ptr, 1, r_ptr, 1);
#else
    // Otherwise we do it the old fashioned way
    // TODO : find methods for elementwise division on other platforms
    for(size_t i=0; i<size; ++i) {
        result.data[i] = data[i] / rhs.data[i];
    }
#endif
    return result;
}

Tensor Tensor::operator*(double rhs) const {
    Tensor result(shape);
    result = *this;
    auto size_i = static_cast<int>(size);
    cblas_dscal(size_i, rhs, result.data.get(), 1);
    return result;
}

Tensor Tensor::operator%(const Tensor& rhs) const {
    if(shape != rhs.shape) throw invalid_argument(size_err);
    Tensor result(shape);
    auto r_ptr = result.data.get();
    auto rhs_ptr = rhs.data.get();
    auto t_ptr = data.get();
    auto size_i = static_cast<int>(size);
    cblas_dsbmv(CblasRowMajor, CblasLower, size_i, 0, 1, t_ptr, 1, rhs_ptr, 1, 1, r_ptr, 1);
    return result;
}

void Tensor::operator+=(const Tensor& rhs) {
    *this = *this + rhs;
}
void Tensor::operator-=(const Tensor& rhs) {
    *this = *this - rhs;
}
void Tensor::operator*=(const Tensor& rhs) {
    *this = *this * rhs;
}
void Tensor::operator/=(const Tensor& rhs) {
    *this = *this / rhs;
}
void Tensor::operator%=(const Tensor& rhs) {
    *this = *this % rhs;
}

Tensor operator+(double lhs, const Tensor& rhs) {
    return rhs + lhs;
}
Tensor operator-(double lhs, const Tensor& rhs) {
    return (rhs * -1) - lhs;
}
Tensor operator*(double lhs, const Tensor& rhs) {
    return rhs * lhs;
}
Tensor operator/(double lhs, const Tensor& rhs) {
#if __APPLE__
    Tensor result(rhs);
    auto size_i = static_cast<int>(rhs.size);
    vvrec(result.data.get(), rhs.data.get(), &size_i);
    cblas_dscal(size_i, lhs, result.data.get(), 1);
#else
    Tensor result(rhs.shape);
    for(size_t =0 ; i < result.size; ++i) {
        result.data[i] = lhs / rhs.data[i];
    }
#endif
    return result;
}

// Dot product is kept as a friend function might be worth adding as a member
double dot(const Tensor& lhs, const Tensor& rhs) {
    if(lhs.shape != rhs.shape) throw invalid_argument(size_err);
    auto size_i = static_cast<int>(lhs.size);
    return cblas_ddot(size_i, lhs.data.get(), 1, rhs.data.get(), 1);
}

Tensor Tensor::t() const {
    Tensor result(shape[1], shape[0]);
    vDSP_mtransD(data.get(), 1,result.data.get(), 1, shape[0], shape[1]);
    return result;
}

std::ostream& operator<<(std::ostream& os, const Tensor& rhs) {
    for(size_t i=0; i<rhs.size; ++i) {
        os << " ";
        os << rhs.data[i];
        if((i+1)%rhs.shape[0] == 0) {
            os << "\n";
        }
    }
    return os;
}

// Initializers
void Tensor::rand(double min, double max) {
    std::uniform_real_distribution<> rand_r(min, max);
    for(auto &it : *this) {
        it = rand_r(eng);
    }
}

void Tensor::rand_int(int min, int max) {
    std::uniform_int_distribution<> i_rand(min, max);
    for(auto &it: *this) {
        it = i_rand(eng);
    }
}

void Tensor::randn(double mean, double var) {
    std::normal_distribution<> n_rand(mean, var);
    for(auto &it : *this) {
        it = n_rand(eng);
    }
}

void Tensor::ones() {
    auto size_i = static_cast<int>(size);
    catlas_dset(size_i, 1, data.get(), 1);
}

void Tensor::zeros() {
    auto size_i = static_cast<int>(size);
    catlas_dset(size_i, 0.0, data.get(), 1);
}
void Tensor::constant(double init) {
    auto size_i = static_cast<int>(size);
    catlas_dset(size_i, init, data.get(), 1);
}
} // namespace nn
