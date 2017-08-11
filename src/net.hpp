#ifndef NET_H
#define NET_H
#include "autodiff.hpp"
#include <stack>

namespace nn{
class Net{
    public:
    Net(double=0.01);
    void backwardProp(const autodiff::var &loss);
    void update();
    class Parameter : public autodiff::var{
        public:
        Parameter(Net* net, const Tensor& data) 
        : autodiff::var(data){ net->param_list.push(this); } 
        Parameter(Net* net ,size_t x, size_t y=1, size_t z=1, size_t t=1)
        : autodiff::var(x,y,z,t){ net->param_list.push(this); }
    };
    private:
    double learning_rate;
    protected:
    std::stack<Parameter*> param_list;
};
} // namespace nn
#endif // NET_H
