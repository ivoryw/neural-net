#ifndef NET_H
#define NET_H
#include "autodiff.hpp"
#include <forward_list>

namespace nn{
class Net{
    public:
    void backward(const autodiff::Var &loss);
    class Parameter : public autodiff::Var{
    public:
        Parameter(Net* net, const Tensor& data) 
        : autodiff::Var(data){ net->param_list.push_front(this); }
        Parameter(Net* net ,size_t x, size_t y=1, size_t z=1, size_t t=1)
        : autodiff::Var(x,y,z,t){ net->param_list.push_front(this); }
    };
    std::forward_list<Parameter*>& params() {return param_list;}
    protected:
    std::forward_list<Parameter*> param_list;
};
} // namespace nn
#endif // NET_H
