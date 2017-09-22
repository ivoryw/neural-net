#ifndef LAYER_H
#define LAYER_H
#include "autodiff.hpp"
#include "net.hpp"

namespace nn{
using autodiff::Var;

    class FullyConnected{
    private:
    Var &weight, &bias;
    public:
        FullyConnected(size_t, size_t, Net*);
        Var operator()(const Var&);
    };

//   class Conv1d{
//   private:
//       Net::Parameter weight, bias;
//       const size_t out_channels, kernel, padding, stride;
//   public:
//       Conv1d(Net*, size_t, size_t, size_t, size_t=0, size_t=1);
//       autodiff::Var operator()(const autodiff::Var&);
//   };

//   class Conv2d{
//   private:
//       Net::Parameter weight, bias;
//       const size_t out_channels, kernel, padding, stride;
//   public:
//       Conv2d(Net*, size_t, size_t, size_t, size_t=0, size_t=1);
//       autodiff::Var operator()(const autodiff::Var&);
//   };

} // namespace nn
#endif // LAYER_H
