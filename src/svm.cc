#include <svm.h>

namespace svm{
SVM::SVM(size_t i_max_, Kernel& kernel_, double C_, double epsilon_)
: kernel(kernel_), i_max(i_max), C(C_), epsilon(epsilon_){}

void SVM::fit(const nn::Tensor& X, const nn::Tensor& y){
    auto n = X.shape[0];
    auto d = X.shape[1];
    auto alpha = nn::Tensor(1,n);
    alpha.zeros();
    for(size_t i=0; i<max_iter; ++i){
        auto alpha_prev = alpha;
        smo();
        if(diff < epsilon){
            break;
        }
    }
}

nn::Tensor SVM::predict(const nn::Tensor& h){
    return sign(nn::dot(w.t(), X.t() + b));
}

void SVM::smo(){
    for(size_t j=0; j<n){
        auto i = rand_int(0, n-1, j);
        auto x_i = X.col(i);
        auto x_j = X.col(j);
        auto y_i = y(i);
        auto y_j = y(j);
        auto k_ij = kernel(x_i, x_i) + kernel(x_j, x_j);
        if (k_ij == 0){
            continue;
        }
        auto alpha_1_i = alpha[i];
        auto alpha_1_j = alpha[j];
        L = lagrange();
        h = 
    }
}

LinearKernel::LinearKernel() : Kernel(){};
nn::Tensor LinearKernel::operator()(const nn::Tensor& a, const nn::Tensor& b){
    return nn::dot(a,b.t());
}

PolyKernel::PolyKernel(size_t degree_) : Kernel(), degree(degree_){}
nn::Tensor PolyKernel::operator()(const nn::Tensor& a, nn::Tensor& b){
    return nn::pow(nn::dot(a, b.t()), degree);
}
}
