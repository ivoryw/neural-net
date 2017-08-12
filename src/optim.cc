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

Moment::Moment(ParStack& p_list, double l_rate_, double moment_)
: Opt(p_list), l_rate(l_rate_), moment(moment_){
    for(auto &it : param_list){
        m_list.emplace_front(nn::Tensor(it->data.shape, 0));
    }
}

void Moment::step(){
    auto p_it = param_list.begin();
    for(auto &m_it : m_list){
        m_it = moment * m_it + l_rate * (*p_it)->grad();
        **p_it -= m_it;
        ++p_it;
    }
}
} // namespace opt
