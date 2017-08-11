/***************************************************
 AutoGrad
 A small library for handling automatic differentiation using a tape
 method. Operator overloads and overloads for standard cmath functions
 are provided so that variables van be used normally until the gradient
 needs to be evaluated.
 ***************************************************/

#include "autodiff.hpp"

#include <utility>
#include <vector>
#include <iostream>

using std::array;
using std::vector;

enum OpType{scalar, matmul};

struct Node{
    array<size_t, 2> parents;
    array<nn::Tensor, 2> weights;
    OpType type;
    
    Node(size_t, size_t, const nn::Tensor&, const nn::Tensor&, OpType);
    Node(size_t, const nn::Tensor&);
    Node();
};

Node::Node(size_t x_index, size_t y_index, const nn::Tensor &x_val, const nn::Tensor &y_val, OpType type_)
: parents{{x_index, y_index}}, weights{{x_val, y_val}}, type(type_){}

Node::Node(size_t index , const nn::Tensor &val)
: parents{{index, 0}}, weights{{val, nn::Tensor()}}, type(scalar){}

Node::Node()
: parents{{0,0}}, weights{{nn::Tensor(), nn::Tensor()}}, type(scalar){}

/**************************************************
 Tape
 A ticker tape which holds the nodes for each operation on an autodiff::var.
 Currently contains the vector for the gradient wrt to the last last node
 on each tree.
 Takes binary, unary and nullary operations.
 ***************************************************/

struct Tape{
    vector<Node> nodes;
    vector<nn::Tensor> grads;
    
    size_t push_0(){
        auto size = nodes.size();
        nodes.emplace_back(Node(size, size, nn::Tensor(), nn::Tensor(), scalar));
        return size;
    }
    size_t push_1(size_t x_index, const nn::Tensor &data){
        auto size = nodes.size();
        nodes.emplace_back(Node(x_index, data));
        return size;
    }
    size_t push_2(size_t x_index, size_t y_index, const nn::Tensor &x_val, const nn::Tensor &y_val, OpType type){
        auto size = nodes.size();
        nodes.emplace_back(Node(x_index, y_index, x_val, y_val, type));
        return size;
    }
    void push_grad(const nn::Shape& shape){
        grads.emplace_back(nn::Tensor(shape,0));
    }
    
    size_t size(){ return nodes.size(); }

    auto begin(){ return nodes.begin(); }
    auto end(){ return nodes.end(); }
};

static Tape tape;

/***************************************************
 var
 Every operation is pushed back onto the ticker tape.
 Holds its own index on the tape along with a reference to the tape.
 ***************************************************/
namespace autodiff{

// Constructor for intializing variable with a tape entry
var::var(const nn::Tensor& data_, size_t index_) :index(index_), data(data_){}

// Constructor for intializing a variable without a tape entry
var::var(const nn::Tensor& data_) : data(data_){
    index = tape.push_0(); 
    tape.push_grad(data.shape);
}

var::var(size_t x, size_t y, size_t z, size_t t)
: data(nn::Tensor(x,y,z,t)){
    index = tape.push_0();
    tape.push_grad(data.shape);
}

void var::evaluateLeaves()const{
    tape.grads[index].ones();
    for(size_t i=index+1; i-- >0;){
        auto &gradient = tape.grads[i];
        auto &node = tape.nodes[i];
        auto &g_shape = gradient.shape;
        auto &w1_shape = node.weights[0].shape;
        auto &w2_shape = node.weights[1].shape;
        if(node.type == matmul){
            tape.grads[node.parents[0]] +=  gradient * node.weights[0].t();
            tape.grads[node.parents[1]] += node.weights[1].t() * gradient;
        }
        else if(node.type == scalar){
            if(w1_shape == g_shape){
                tape.grads[node.parents[0]] += gradient % node.weights[0];
            }
            if(w2_shape == g_shape){
                tape.grads[node.parents[1]] += gradient % node.weights[1];
            }
        }
    } 
}

nn::Tensor var::grad()const{ return tape.grads[index]; }        

double& var::operator()(size_t x, size_t y, size_t z, size_t t){
    return data(x,y,z,t);
}

var var::operator+(const var& y)const{
    auto new_data = data + y.data;
    auto x_weight = nn::Tensor(data.shape, 1);
    auto y_weight = nn::Tensor(data.shape, 1);
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return var(new_data, new_index);
}
var var::operator-(const var& y)const{
    auto new_data = data - y.data;
    auto x_weight = nn::Tensor(data.shape, 1);
    auto y_weight = nn::Tensor(data.shape, -1);
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return var(new_data, new_index);
}
var var::operator%(const var& y)const{
    auto new_data = data % y.data;
    auto x_weight = y.data; 
    auto y_weight = data;
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return var(new_data, new_index);
}
var var::operator/(const var& y)const{
    auto x_weight = 1.0 / y.data;
    auto y_weight = 2.0 * data / (y.data % y.data);
    auto new_data = data / y.data;
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return var(new_data, new_index);
}

var var::operator*(const var& y)const{
    auto x_weight = y.data;
    auto y_weight = data;
    auto new_data = data * y.data;
    auto new_index = tape.push_2(index, y.index, x_weight, y_weight, matmul);
    tape.push_grad(new_data.shape);
    return var(new_data, new_index);
}


void var::operator=(const nn::Tensor& y) { data = y;}

void var::operator+=(const var& y){ *this = *this + y;}
void var::operator-=(const var& y){ *this = *this - y;}
void var::operator/=(const var& y){ *this = *this/y; }
void var::operator*=(const var& y){ *this = *this * y; }
void var::operator%=(const var& y){ *this = *this % y; }

std::ostream& operator<<(std::ostream& os, const var& rhs){
    os << "Tensor:" << std::endl;
    os << rhs.data;
    os << "Gradient:" << std::endl;
    os << tape.grads[rhs.index];
    os << std::endl;
    return os;
}
/***************************************************
cmath arithmetics
Reimplimentation of primitive cmath operations to include
differentiation process.
***************************************************/

var pow(const var &x, double y){
    auto x_weight = y * nn::pow(x.data, y - 1);
    auto new_index = tape.push_1(x.index, x_weight);
    return var(pow(x.data, y), new_index);
}
var pow(const var &x, const var &y){
    auto x_weight = y.data % nn::pow(x.data, y.data-1);
    auto pow_x_y = nn::pow(x.data,y.data);
    auto y_weight = pow_x_y * nn::log(x.data);
    auto new_index = tape.push_2(x.index, y.index, x_weight, y_weight, scalar);
    return var(pow_x_y, new_index);
}

//var sqrt(const var &x) {
//    auto new_index = tape.push_1(x.index, sqrt(x.data) * 0.5 );
//    return var(sqrt(x.data), new_index);
//}
//
//var exp(const var &x){
//    auto exp_x = tape.push_1(x.index, x.data * nn::exp(x.data));
//    auto new_index = exp_x;
//    return var(exp_x, new_index);
//}
//
//var log(const var &x){
//    auto new_index = tape.push_1(x.index, 1/x.data);
//    return var(log(x.data), new_index);
//}
//
var sin(const var &x){
    auto new_index = tape.push_1(x.index, nn::cos(x.data));
    return var(nn::sin(x.data), new_index);
}

var cos(const var &x){
    auto new_index = tape.push_1(x.index, nn::sin(x.data) * -1.0);
    return var(nn::cos(x.data), new_index);
}

var tan(const var &x){
    auto cos_x = nn::cos(x.data);
    auto new_index = tape.push_1(x.index, 1.0 / (cos_x * cos_x));
    return var(nn::tan(x.data), new_index);
}

var asin(const var &x){
    auto weight = 1.0 / nn::sqrt(1.0 - x.data * x.data);
    auto new_index = tape.push_1(x.index, weight);
    return var(nn::asin(x.data), new_index);
}

var acos(const var &x){
    auto new_index = tape.push_1(x.index, -1.0 / nn::sqrt(1.0 - x.data * x.data));
    return var(nn::acos(x.data), new_index);
}

var atan(const var &x){
    auto new_index = tape.push_1(x.index, 1.0 / (1.0 + x.data * x.data));
    return var(atan(x.data), new_index);
}

var conv1d(const var& x, const var& y){
    auto new_data= nn::conv1d(x.data, y.data);
    auto x_weight = nn::conv1d(nn::Tensor(x.data.shape,1), y.data);
    auto y_weight = nn::conv1d(x.data, nn::Tensor(y.data.shape,1));
    auto new_index = tape.push_2(x.index, y.index, x_weight, y_weight, scalar);
    tape.push_grad(new_data.shape);
    return var(new_data, new_index);
}

//var conv2d(const var& x, const var& weight, size_t stride, size_t kernel){
//    auto new_value = nn::conv2d(x.data, weight.data, stride, kernel);
//    auto x_weight = weight.data;
//    auto y_weight = x.data;
//    auto new_index = tape.push_2(x.index, weight.index, x_weight, y_weight, conv);
//    return var(new_value, new_index);
//}

} // namespace autodiff
