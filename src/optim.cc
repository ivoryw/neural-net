#include "optim.hpp"

namespace opt{
Opt::Opt(ParameterList& parameters_)
: parameters(parameters_){}

GD::GD(ParameterList& parameters, double l_rate_)
: Opt(parameters), l_rate(l_rate_){}

void GD::step(){
    for(auto &it : parameters){
        auto parameter_update = it->grad() * l_rate;
        *it -= parameter_update;
    }
}

Moment::Moment(ParameterList& parameters, double l_rate_, double moment_)
: Opt(parameters), l_rate(l_rate_), moment(moment_){
    for(auto &it : parameters){
        m_list.push_back(nn::Tensor(it->data.shape, 0));
    }
}

void Moment::step(){
    auto p_it = parameters.begin();
    for(auto &m_it : m_list){
        m_it = moment * m_it + l_rate * (*p_it)->grad();
        **p_it -= m_it;
        ++p_it;
    }
}
} // namespace opt
