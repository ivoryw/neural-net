#ifndef SVM_H
#define SVM_H
#include "tensor.hpp"

namespace svm{
class SVM{
public:
    SVM(size_t, const Kernel&, double, double);
    void fit(const nn::Tensor& X, const nn::Tensor& y);
    nn::Tensor predict(const nn::Tensor& X);
private:
    const size_t i_max;
    const Kernel kernel;
    const double C_, epsilon_;
    nn::Tensor w, b;
    void smo();
}
} // namespace svm
#endif // SVM_H

class Kernel{
    Kernel();
    virtual nn::Tensor operator()(const nn:Tensor&, const nn::Tensor&);
}

class LinearKernel : public Kernel{
    LinearKernel();
}

class PolyKernel : public Kernel{
    PolyKernel(size_t);
}
