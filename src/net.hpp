#ifndef NET_H
#define NET_H
#include "autodiff.hpp"
#include <forward_list>

namespace nn{
using autodiff::Var;

class Net{    
    public:
    void backward(const Var &loss);
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
