#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.hpp"

namespace autodiff{
    
    class Var;
    
    Var pow(const Var&, double);
    Var pow(const Var&, const Var&);
    Var sqrt(const Var&);
    Var exp(const Var&);
    Var log(const Var&);
    Var sin(const Var&);
    Var cos(const Var&);
    Var tan(const Var&);
    Var asin(const Var&);
    Var acos(const Var&);
    Var atan(const Var&);
    Var conv_1d(const Var&, const Var&);
    Var conv_2d(const Var&, const Var&, size_t, size_t);
    
class Var{
    size_t index;

public:
    nn::Tensor data;
    Var(const nn::Tensor&);
    Var(const nn::Tensor&, size_t);
    Var(size_t, size_t=1, size_t=1, size_t=1);
    
    void evaluate_leaves() const;
    nn::Tensor grad()const;

    double& operator()(size_t, size_t=0, size_t=0, size_t=0);
    double operator()(size_t, size_t=0, size_t=0, size_t=0) const;

    Var operator+(const Var&)const;
    Var operator-(const Var&)const;
    Var operator*(const Var&)const;
    Var operator%(const Var&)const;
    Var operator/(const Var&)const;

    void operator=(const nn::Tensor&);
    void operator+=(const Var&);
    void operator-=(const Var&);
    void operator/=(const Var&);
    void operator*=(const Var&);
    void operator%=(const Var&);

    friend Var operator*(double, const Var&);

    friend std::ostream& operator<<(std::ostream&, const Var&);
    
    friend Var pow(const Var&, double);
    friend Var pow(const Var&, const Var&);
    friend Var sqrt(const Var&);
    friend Var exp(const Var&);
    friend Var log(const Var&);
    friend Var sin(const Var&);
    friend Var cos(const Var&);
    friend Var tan(const Var&);
    friend Var asin(const Var&);
    friend Var acos(const Var&);
    friend Var atan(const Var&);
    friend Var conv_1d(const Var&, const Var&);
    friend Var conv_2d(const Var&, const Var&, size_t, size_t);

    void randn(int mean=0, int var=1){ data.randn(mean, var); }
    Var abs_sum();
    Var sum();

    nn::Tensor::Iterator begin(){ return data.begin(); }
    nn::Tensor::Iterator end(){ return data.end(); }
    size_t size(){ return data.size; }
};
}
#endif //AUTOGRAD_H
