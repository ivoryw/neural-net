#ifndef NET_H
#define NET_H
#include "autodiff.hpp"
#include <forward_list>

namespace nn{
using autodiff::Var;

/**
	Net
	An abstract neural net object using automatic differentiation
	A forward function should be defined by the end-user within a child class
*/
class Net{    
    public:
    // Backpropigation using the chain rule and the AutoDiff module
    void backward(const Var &loss);
    // Parameter registration and creation
    Var& create_parameter(const Tensor& data) {
        parameters.push_front(Var(data));
        return parameters.front();
    }
    std::forward_list<Var>& params() { return parameters; }
    protected:
    std::forward_list<Var> parameters;
};
} // namespace nn
#endif // NET_H
