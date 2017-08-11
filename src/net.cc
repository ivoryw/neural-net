#include "net.hpp"

namespace nn{
Net::Net(double lr) : learning_rate(lr){}

void Net::backwardProp(const autodiff::var& loss){
    loss.evaluateLeaves();
}

void Net::update(){
    while(!param_list.empty()){
        auto param_update = param_list.top()->grad() * learning_rate;
        *(param_list.top()) += param_update;
        param_list.pop();
    }
}
} // namespace nn
