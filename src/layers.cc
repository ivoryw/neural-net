#include "layers.hpp"
#include <string>
#include <cmath>

namespace nn{

using autodiff::Var;

static std::string dim_err ="Input tensors to fully connected layers must be 1D tensors";

FullyConnected::FullyConnected(size_t in_size, size_t out_size, Net* net)
: weight(net->create_parameter(Tensor(in_size, out_size))),
  bias (net->create_parameter(Tensor(1, out_size))) {
    weight.randn();
    bias.randn();
}
    
Var FullyConnected::operator()(const Var& input){
    auto result = weight * input;
    result += bias;
    return result;
}

Conv1d::Conv1d(Net* net, size_t c_in, size_t c_out, size_t kernel, size_t padding)
: weight(net->create_parameter(Tensor(c_out, c_in, kernel))),
bias(net->create_parameter(Tensor(c_out))),
out_channels(c_out), kernel(kernel), padding(padding) {
    weight.randn();
    bias.randn();
}

Var Conv1d::operator()(const autodiff::Var& input) {
    auto padded_input = pad(input);
    auto r_len = floor((padded_input.data.shape[1] + 2*padding - (kernel - 1) -1)/1);   
    Var result(out_channels, r_len);
    for(size_t i = 0; i < result.data.shape[2]; ++i) {
        result.set_row(i,conv1d(weight.col(i),input));
        result.set_row(i, result.row(i)+bias(i));
   	}
    result += bias;
    return result;
}

Var Conv1d::pad(const Var& input) {
    Var padded(input.data.shape[0] + padding*2, input.data.shape[1]);
    for(size_t i = 0; i < padded.data.shape[0]; ++i) {
        if(i < padding || i >= padded.data.shape[0] - padding) {
            padded.set_row(i, nn::Tensor({{padded.data.shape[0], 1, 1}}, 0));
        }
        else{
            padded.set_row(i, input.row(i - padding));
        }
    }
    return padded;
}


//Conv2d::Conv2d(Net* net, size_t c_in, size_t c_out, size_t kernel, size_t padding, size_t stride)
//: weight(net, c_out, c_in, kernel, kernel), bias(net, c_out), 
//out_channels(c_out), kernel(kernel), padding(padding), stride(stride){
//    weight.randn();
//    bias.randn();
//}
//
//autodiff::var Conv2d::operator()(const autodiff::var& input){
//    auto r_height = floor((input.data.shape[1] + 2*padding - (kernel - 1) -1)/stride + 1);
//    auto r_width = floor((input.data.shape[0] + 2*padding - (kernel - 1) - 1)/stride + 1);
//    autodiff::var result(out_channels, r_width, r_height);
//    result = conv2d(input, weight, stride, kernel); 
//    result += bias;
//    return result;
//}
} // namespace nn
