#include "autodiff.hpp"
#include "net.hpp"
#include "layers.hpp"

#include <iostream>

class Model : public nn::Net{
    public:
    Model() : nn::Net(), fc1(10, 5, this), fc2(5,1,this){ }

    nn::fc fc1, fc2;
    autodiff::var forward(autodiff::var& x) {
        auto y = fc1(x);
        auto z = fc2(y);
        return z;
    }
};

int main(){
    nn::Tensor a(1,10);
    a.randn();
    autodiff::var var_a(a);

    auto model = Model();
    auto y = model.forward(var_a);
    model.backwardProp(y);
    model.update();
    y = model.forward(var_a);
    std::cout << var_a;
    
    nn::Tensor b(1,10);
    b.randn();
    auto d = autodiff::var(b);
    auto c = autodiff::conv1d(d,var_a);
    c.evaluateLeaves();
    std::cout << c;
    std::cout << d;
}
