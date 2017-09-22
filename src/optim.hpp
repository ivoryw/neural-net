#ifndef OPTIM_H
#define OPTIM_H

#include "net.hpp"
#include <list>

namespace opt{
typedef std::forward_list<nn::Net::Parameter*> ParameterList;
class Opt{
public:
    Opt(ParameterList&);
    virtual void step(){}
    virtual ~Opt(){}
protected:
    ParameterList parameters;
};

class GD: public Opt{
public:
    GD(ParameterList&, double = 0.1);
    void step();    
    ~GD(){};
private:
    double l_rate;
};

class Moment : public Opt{
public:
    Moment(ParameterList&, double = 0.1, double = 0.9);
    void step();
    ~Moment(){};
private:
    double l_rate, moment;
    std::list<nn::Tensor> m_list;
};

} // namespace opt
#endif // OPTIM_H
