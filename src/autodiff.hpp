#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.hpp"

namespace autodiff{
    
    class var;
    
    var pow(const var&, double);
    var pow(const var&, const var&);
    var sqrt(const var&);
    var exp(const var&);
    var log(const var&);
    var sin(const var&);
    var cos(const var&);
    var tan(const var&);
    var asin(const var&);
    var acos(const var&);
    var atan(const var&);
    var conv1d(const var&, const var&);
    var conv2d(const var&, const var&, size_t, size_t);
    
class var{
private:
    size_t index;

public:
    nn::Tensor data;
    var(const nn::Tensor&);
    var(const nn::Tensor&, size_t);
    var(size_t, size_t=1, size_t=1, size_t=1);
    
    void evaluateLeaves() const;
    nn::Tensor grad()const;

    double& operator()(size_t,size_t=1,size_t=1,size_t=1);

    var operator+(const var&)const;
    var operator-(const var&)const;
    var operator*(const var&)const;
    var operator%(const var&)const;
    var operator/(const var&)const;

    void operator=(const nn::Tensor&);
    void operator+=(const var&);
    void operator-=(const var&);
    void operator/=(const var&);
    void operator*=(const var&);
    void operator%=(const var&);

    friend std::ostream& operator<<(std::ostream&, const var&);
    
    friend var pow(const var&, double);
    friend var pow(const var&, const var&);
    friend var sqrt(const var&);
    friend var exp(const var&);
    friend var log(const var&);
    friend var sin(const var&);
    friend var cos(const var&);
    friend var tan(const var&);
    friend var asin(const var&);
    friend var acos(const var&);
    friend var atan(const var&);
    friend var conv1d(const var&, const var&);
    friend var conv2d(const var&, const var&, size_t, size_t);

    void randn(int mean=0, int var=1){ data.randn(mean, var); }
    var asum();

    nn::Tensor::iterator begin(){ return data.begin(); }
    nn::Tensor::iterator end(){ return data.end(); }
    size_t size(){ return data.size; }
};
}
#endif //AUTOGRAD_H
