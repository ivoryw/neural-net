#include "autodiff.hpp"
#include "net.hpp"
#include "layers.hpp"
#include "optim.hpp"
#include "loss.hpp"

#include <iostream>

class Model : public nn::Net{
    public:
    Model() : nn::Net(), fc1(10, 1, this){ }

    nn::FullyConnected fc1;
    autodiff::Var forward(autodiff::Var& x) {
        auto y = fc1(x);
        return y;
    }
};

int main(){
    nn::Tensor a(1,10);
    nn::Tensor z(1);
    a.rand(0,1);
    z.rand(0,1);
    std::cout << a.row(0);
    std::cout << a.col(0);
    autodiff::Var var_a(a);
    autodiff::Var var_z(z);

    Model model;
    auto opt = opt::GD(model.params());
    auto y = model.forward(var_a);
    auto l = loss::cross_entropy_loss(y, var_z);
    model.backward(y);
    opt.step();
    std::cout << l;
}
