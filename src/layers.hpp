#ifndef LAYER_H
#define LAYER_H
#include "autodiff.hpp"
#include "net.hpp"

namespace nn{
    class fc{
    private:
        Net::Parameter weight, bias;
    public:
        fc(size_t, size_t, Net*);
        autodiff::var operator()(const autodiff::var&);
    };

   class Conv1d{
   private:
       Net::Parameter weight, bias;
       const size_t out_channels, kernel, padding, stride;
   public:
       Conv1d(Net*, size_t, size_t, size_t, size_t=0, size_t=1);
       autodiff::var operator()(const autodiff::var&);
   };

   class Conv2d{
   private:
       Net::Parameter weight, bias;
       const size_t out_channels, kernel, padding, stride;
   public:
       Conv2d(Net*, size_t, size_t, size_t, size_t=0, size_t=1);
       autodiff::var operator()(const autodiff::var&);
   };

} // namespace nn
#endif // LAYER_H
