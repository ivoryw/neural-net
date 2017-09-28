#ifndef OPTIM_H
#define OPTIM_H

#include "net.hpp"
#include <list>

namespace opt{

using autodiff::Var;
typedef std::forward_list<Var> ParameterList;

/**
    Opt
    Abstract optimiser storing a reference to a list of autodiff::Vars
*/
class Opt{
public:
    Opt(ParameterList&);
    virtual void step(){}
    virtual ~Opt(){}
protected:
    ParameterList& parameters;
};

/**
    GD
    Gradient descent optimisation of a ParameterList
*/
class GD: public Opt{
public:
    GD(ParameterList&, double = 0.1);
    // One optimisation step
    void step();    
    ~GD(){};
private:
    double l_rate;
};

/**
    Moment
    Gradient descent optimisation of a ParameterList with momentum
*/
class Moment : public Opt{
public:
    Moment(ParameterList&, double = 0.1, double = 0.9);
    // One optimisation step
    void step();
    ~Moment(){}
private:
    double l_rate, moment;
    std::list<nn::Tensor> m_list;
};

} // namespace opt
#endif // OPTIM_H
