#include "tensor.hpp"

#include <cmath>
#include <string>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif // __APPLE__

namespace nn{
static std::string size_err {"Tensor sizes do not match"};
    
using std::invalid_argument;

Tensor sin(const Tensor& rhs){
    Tensor result(rhs.shape);
#ifdef __APPLE__
    auto size_ = static_cast<int>(result.size);
    vvsin(result.data.get(), rhs.data.get(), &size_);
#else
    for(size_t i =0 ; i < result.size; ++i){
        result.data[i] = sin(rhs.data[i]);
    }
#endif
    return result;
}

Tensor cos(const Tensor& rhs){
    Tensor result(rhs.shape);
#ifdef __APPLE__
    auto size_ = static_cast<int>(result.size);
    vvcos(result.data.get(), rhs.data.get(), &size_);
#else
    for(size_t i=0 ; i< result.size; ++i){
        result.data[i] = cos(rhs.data[i]);
    }
#endif
    return result;
}

Tensor tan(const Tensor& rhs){
    Tensor result(rhs.shape);
#ifdef __APPLE__
    auto size_ = static_cast<int>(result.size);
    vvtan(result.data.get(), rhs.data.get(), &size_);
#else
    for(size_t i=0; i<result.size; ++i){
        result.data[i] = tan(rhs.data[i]);
    }
#endif
    return result;
}

Tensor asin(const Tensor& rhs){
    Tensor result(rhs.shape);
#ifdef __APPLE__
    auto size_ = static_cast<int>(result.size);
    vvasin(result.data.get(), rhs.data.get(), &size_);
#else
    for(size_t i=0; i<result.size; ++i){
        result.data[i] = asin(rhs.data[i]);
    }
#endif
    return result;
}

Tensor acos(const Tensor& rhs){
    Tensor result(rhs.shape);
#ifdef __APPLE__
    auto size_ = static_cast<int>(result.size);
    vvacos(result.data.get(), rhs.data.get(), &size_);
#else
    for(size_t i=0; i<result.size; ++i){
        result.data[i] = acos(rhs.data[i];);
    }
#endif
    return result;
}

Tensor atan(const Tensor& rhs){
    Tensor result(rhs.shape);
#ifdef __APPLE__
    auto size_ = static_cast<int>(result.size);
    vvatan(result.data.get(), rhs.data.get(), &size_);
#else
    for(size_t i=0; i<result.size; ++i){
        result.data[i] = atan(rhs.data[i]);
    }
#endif
    return result;
}

Tensor pow(const Tensor& base, double power){
    Tensor result(base.shape);
    Tensor pow_tens(base.shape,power);
#ifdef __APPLE__
    auto size_ = static_cast<int>(base.size);
    vvpow(result.data.get(), base.data.get(), pow_tens.data.get(), &size_);
#endif
    return result;
}

Tensor pow(const Tensor& base, const Tensor& power){
    if(base.shape != power.shape) throw invalid_argument(size_err);
    Tensor result(base.shape);
#ifdef __APPLE__
    auto size_ = static_cast<int>(base.size);
    vvpow(result.data.get(), base.data.get(), power.data.get(), &size_);
#endif
    return result;
}

Tensor sqrt(const Tensor& base){
    Tensor result(base.shape);
    auto size_ = static_cast<int>(base.size);
#ifdef __APPLE__
    vvsqrt(result.data.get(), base.data.get(), &size_);
#endif
    return result;
}

Tensor log(const Tensor& base){
    Tensor result(base.shape);
    auto size_ = static_cast<int>(base.size);
#ifdef __APPLE__
    vvlog(result.data.get(), base.data.get(), &size_);
#endif
    return result;
}
}
