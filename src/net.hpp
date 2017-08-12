#ifndef NET_H
#define NET_H
#include "autodiff.hpp"
#include <forward_list>

namespace nn{
class Net{
public:
    void backwardProp(const autodiff::var &loss);
    class Parameter : public autodiff::var{
        public:
        Parameter(Net* net, const Tensor& data) 
        : autodiff::var(data){ net->param_list.push_front(this); } 
        Parameter(Net* net, size_t x, size_t y=1, size_t z=1, size_t t=1)
        : autodiff::var(x,y,z,t){ net->param_list.push_front(this); }
    };
    std::forward_list<Parameter*>& params() {return param_list;}
protected:
    std::forward_list<Parameter*> param_list;
};
} // namespace nn
#endif // NET_H
