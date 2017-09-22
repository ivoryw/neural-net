#include "autodiff.hpp"

namespace loss{

using autodiff::Var;

Var l1_loss(Var&, Var&);
Var mean_error(Var&, Var&);
Var mean_squared_error(Var&, Var&);
Var cross_entropy_loss(Var&, Var&);
} // namepsace loss

