#include "autodiff.hpp"

#include <utility>
#include <vector>
#include <iostream>

using std::array;
using std::vector;
using nn::Tensor;

enum OpType{scalar, matmul, reduct};

/**
    WegnerntNode
    A node holding weights for variable parents
*/
struct WegnerntNode{
    array<size_t, 2> parents;
    array<Tensor, 2> weights;
    OpType type;
    
    WegnerntNode(size_t, size_t, const Tensor&, const Tensor&, OpType);
    WegnerntNode(size_t, const Tensor&, OpType=scalar);
    WegnerntNode();
};

//  Two parent variable initialiser 
WegnerntNode::WegnerntNode(size_t x_index, size_t y_index, const Tensor &x_weight, const Tensor &y_weight, OpType type_)
: parents{{x_index, y_index}}, weights{{x_weight, y_weight}}, type(type_){}

// One parent variable initialiser
WegnerntNode::WegnerntNode(size_t index , const Tensor &weight, OpType type_)
: parents{{index, 0}}, weights{{weight, Tensor()}}, type(type_){}

// Zero parent variable intialiser
WegnerntNode::WegnerntNode()
: parents{{0,0}}, weights{{Tensor(), Tensor()}}, type(scalar){}

/**
    WengerntList
    A ticker tape which holds the nodes for each operation on a Variable
*/
struct WengerntList{
    vector<WegnerntNode> nodes;
    vector<Tensor> grads;

    // Appends a zero parent variable to the tape
    size_t push_0(){
        auto size = nodes.size();
        nodes.emplace_back(WegnerntNode(size, size, Tensor(), Tensor(), scalar));
        return size;
    }

    // Appends a one parent variable to the tape
    size_t push_1(size_t x_index, const Tensor &weight, OpType type=scalar){
        auto size = nodes.size();
        nodes.emplace_back(WegnerntNode(x_index, weight, type));
        return size;
    }

    // Appends a two parent variable to the tape
    size_t push_2(size_t x_index, size_t y_index, const Tensor &x_weight, const Tensor &y_weight, OpType type) {
        auto size = nodes.size();
        nodes.emplace_back(WegnerntNode(x_index, y_index, x_weight, y_weight, type));
        return size;
    }

    // Adds a gradient to the gradient tape
    void push_grad(const nn::Shape& shape) {
        grads.emplace_back(Tensor(shape,0));
    }
    
    size_t size() { return nodes.size(); }

    auto begin() { return nodes.begin(); }
    auto end(){ return nodes.end(); }
};

static WengerntList tape;

namespace autodiff {

Var::Var(const Tensor& data_, size_t index_) :index(index_), data(data_){}

Var::Var(const Tensor& data_) : data(data_){
    index = tape.push_0(); 
    tape.push_grad(data.shape);
}

Var::Var(size_t x, size_t y, size_t z, size_t t) : data(Tensor(x,y,z,t)) {
    index = tape.push_0();
    tape.push_grad(data.shape);
}

// Backpropagates using the chain rule along the leaf nodes
void Var::evaluate_leaves() const {
    tape.grads[index].ones();
    for(size_t i=index+1; i-- >0;){
        auto &gradient = tape.grads[i];
        auto &node = tape.nodes[i];
        auto &g_shape = gradient.shape;
        auto &w1_shape = node.weights[0].shape;
        auto &w2_shape = node.weights[1].shape;
        // Splits due to different multiplication methods
        if(node.type == matmul){
            tape.grads[node.parents[0]] += gradient * node.weights[0].t();
            tape.grads[node.parents[1]] += node.weights[1].t() * gradient;

        }
        else if(node.type == scalar) {
            if(w1_shape == g_shape) {
                tape.grads[node.parents[0]] += gradient % node.weights[0];
            }
            if(w2_shape == g_shape) {
                tape.grads[node.parents[1]] += gradient % node.weights[1];
            }
        }
        else if(node.type == reduct) {
            tape.grads[node.parents[0]] += gradient * node.weights[0];
        }
    } 
}

Tensor Var::grad() const { return tape.grads[index]; }

// Access operators
double& Var::operator()(size_t x, size_t y, size_t z, size_t t) {
    return data(x,y,z,t);
}

double Var::operator()(size_t x, size_t y, size_t z, size_t t) const {
    return data(x,y,z,t);
}

std::ostream& operator<<(std::ostream& os, const Var& rhs) {
    os << "Tensor:" << std::endl;
    os << rhs.data;
    os << "Gradient:" << std::endl;
    os << tape.grads[rhs.index];
    os << std::endl;
    return os;
}

// Differentiation works by assigning weights to the parent variables,
// these weights are evaluated using standard differentiation methods
// A new node is created on the tape with these weights and links to the
// parent variables
Var Var::operator+(const Var& y) const {
    auto new_data = data + y.data;
    auto x_weight = Tensor(data.shape, 1);
    auto y_weight = Tensor(data.shape, 1);
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

Var Var::operator-(const Var& y) const {
    auto new_data = data - y.data;
    auto x_weight = Tensor(data.shape, 1);
    auto y_weight = Tensor(data.shape, -1);
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

Var Var::operator%(const Var& y) const {
    auto new_data = data % y.data;
    auto x_weight = y.data; 
    auto y_weight = data;
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

Var Var::operator/(const Var& y) const {
    auto x_weight = 1.0 / y.data;
    auto y_weight = 2.0 * data / (y.data % y.data);
    auto new_data = data / y.data;
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

Var Var::operator*(const Var& y) const {
    auto x_weight = y.data;
    auto y_weight = data;
    auto new_data = data * y.data;
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, matmul);
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

Var operator*(double x, const Var& y) {
    auto weight = Tensor(y.data.shape, x);
    auto new_data = x * y.data;
    auto new_index = tape.push_1(y.index, weight);
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

void Var::operator=(const Tensor& y) { data = y; }

void Var::operator+=(const Var& y){ *this = *this + y;}
void Var::operator-=(const Var& y){ *this = *this - y;}
void Var::operator/=(const Var& y){ *this = *this / y; }
void Var::operator*=(const Var& y){ *this = *this * y; }
void Var::operator%=(const Var& y){ *this = *this % y; }

Var pow(const Var &x, double y) {
    auto x_weight = y * nn::pow(x.data, y - 1);
    auto new_index = tape.push_1(x.index, x_weight);
    return Var(pow(x.data, y), new_index);
}

Var pow(const Var &x, const Var &y) {
    auto x_weight = y.data % nn::pow(x.data, y.data-1);
    auto pow_x_y = nn::pow(x.data,y.data);
    auto y_weight = pow_x_y * nn::log(x.data);
    auto new_index = tape.push_2(x.index, y.index, x_weight, y_weight, scalar);
    return Var(pow_x_y, new_index);
}

//var sqrt(const var &x) {
//    auto new_index = tape.push_1(x.index, sqrt(x.data) * 0.5 );
//    return var(sqrt(x.data), new_index);
//}
//
//var exp(const var &x) {
//    auto exp_x = tape.push_1(x.index, x.data * nn::exp(x.data));
//    auto new_index = exp_x;
//    return var(exp_x, new_index);
//}
//
Var log(const Var &x) {
    auto new_index = tape.push_1(x.index, 1 / x.data);
    return Var(nn::log(x.data), new_index);
}

Var sin(const Var &x) {
    auto new_index = tape.push_1(x.index, nn::cos(x.data));
    return Var(nn::sin(x.data), new_index);
}

Var cos(const Var &x) {
    auto new_index = tape.push_1(x.index, nn::sin(x.data) * -1.0);
    return Var(nn::cos(x.data), new_index);
}

Var tan(const Var &x) {
    auto cos_x = nn::cos(x.data);
    auto new_index = tape.push_1(x.index, 1.0 / (cos_x * cos_x));
    return Var(nn::tan(x.data), new_index);
}

Var asin(const Var &x) {
    auto weight = 1.0 / nn::sqrt(1.0 - x.data * x.data);
    auto new_index = tape.push_1(x.index, weight);
    return Var(nn::asin(x.data), new_index);
}

Var acos(const Var &x) {
    auto new_index = tape.push_1(x.index, -1.0 / nn::sqrt(1.0 - x.data * x.data));
    return Var(nn::acos(x.data), new_index);
}

Var atan(const Var &x) {
    auto new_index = tape.push_1(x.index, 1.0 / (1.0 + x.data * x.data));
    return Var(atan(x.data), new_index);
}

Var Var::abs_sum() {
    auto new_index = tape.push_1(index, Tensor(data.shape, 1), reduct);
    auto double_new_data = data.abs_sum();
    Tensor new_data(1);
    new_data(0) = double_new_data;
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

Var Var::sum() {
    auto new_index = tape.push_1(index, Tensor(data.shape, 1), reduct);
    auto double_new_data = data.sum();
    Tensor new_data(1);
    new_data(0) = double_new_data;
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

Var conv_1d(const Var& x, const Var& y) {
    auto new_data = nn::conv_1d(x.data, y.data);
    auto x_weight = nn::conv_1d(Tensor(x.data.shape, 1), y.data);
    auto y_weight = nn::conv_1d(x.data, Tensor(y.data.shape, 1));
    auto new_index = tape.push_2(x.index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return Var(new_data, new_index);
}

//var conv2d(const Var& x, const Var& weight, size_t stride, size_t kernel) {
//    auto new_value = nn::conv2d(x.data, weight.data, stride, kernel);
//    auto x_weight = weight.data;
//    auto y_weight = x.data;
//    auto new_index = tape.push_2(x.index, weight.index, x_weight, y_weight, conv);
//    return var(new_value, new_index);
//}

} // namespace autodiff
