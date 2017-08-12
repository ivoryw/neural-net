#ifndef OPTIM_H
#define OPTIM_H

#include "net.hpp"

namespace opt{
typedef std::forward_list<nn::Net::Parameter*> ParStack;
class Opt{
public:
    Opt(ParStack&);
    virtual void step(){}
    virtual ~Opt(){}
protected:
    ParStack param_list;
};

class GD: public Opt{
public:
    GD(ParStack&, double = 0.1);
    void step();    
    ~GD(){};
private:
    double l_rate;
};
} // namespace opt
#endif // OPTIM_H
