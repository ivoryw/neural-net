#include "optim.hpp"

namespace opt{
Opt::Opt(ParStack& param_list_)
: param_list(param_list_){}

GD::GD(ParStack& p_list, double l_rate_) 
: Opt(p_list), l_rate(l_rate_){}

void GD::step(){
    for(auto &it : param_list){
        auto param_update = it->grad() * l_rate;
        *it -= param_update;
    }
}
} // namespace opt
