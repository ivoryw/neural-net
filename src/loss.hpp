#include "autodiff.hpp"

namespace loss{
autodiff::Var mean_error(autodiff::Var&, autodiff::Var&);
autodiff::Var mean_squared_error(autodiff::Var&, autodiff::Var&);
autodiff::Var cross_entropy_loss(autodiff::Var&, autodiff::Var&);
} // namepsace loss

