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

/**
    Var
    An automatically differentiable variable
*/
class Var{
    size_t index;

public:
    nn::Tensor data;

    // Intialisation without an existing tape index
    Var(const nn::Tensor&);

    // Initialisation with an existing tape index
    Var(const nn::Tensor&, size_t);

    // Initialisation with a predefined size
    Var(size_t, size_t=1, size_t=1, size_t=1);
   
    // Evaluate the gradient of all nodes of the tape with respect to self
    void evaluate_leaves() const;

    // Returns the last evaluated gradient
    nn::Tensor grad()const;

    // Element access using a zero index
    double& operator()(size_t, size_t=0, size_t=0, size_t=0);

    // Const element access
    double operator()(size_t, size_t=0, size_t=0, size_t=0) const;

    // Pointwise tensor addition
    Var operator+(const Var&)const;

    // Pointwise tensor subtraction
    Var operator-(const Var&)const;

    // Matrix multiplication
    Var operator*(const Var&)const;

    // Pointwise matrix multiplication
    Var operator%(const Var&)const;
    // Pointwise matrix division
    Var operator/(const Var&)const;

    void operator=(const nn::Tensor&);
    void operator+=(const Var&);
    void operator-=(const Var&);
    void operator/=(const Var&);
    void operator*=(const Var&);
    void operator%=(const Var&);

    // Scalar multiplication
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

    // Initialisation with a normal distibution
    void randn(int mean=0, int var=1){ data.randn(mean, var); }
    Var abs_sum();
    Var sum();

    nn::Tensor::Iterator begin(){ return data.begin(); }
    nn::Tensor::Iterator end(){ return data.end(); }
    size_t size(){ return data.size; }
};
}
#endif //AUTOGRAD_H
