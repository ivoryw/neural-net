#include "tensor.hpp"

#include <memory>
#include <complex>
#include <fftw3.h>

namespace nn{
typedef std::complex<double> xdouble;

void expand_fftw_arr(size_t size, xdouble* array){
    for(size_t i=0; i<size/2; ++i){
        array[size-i-1] = conj(array[i]);
    }
}
    
void fft_r2c_1d(size_t size, double* in_ptr, xdouble* out_ptr){
    auto plan = fftw_plan_dft_r2c_1d(size, in_ptr,
                        reinterpret_cast<fftw_complex*>(out_ptr),
                        FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

void fft_c2r_1d(size_t size, xdouble* in_ptr, double* out_ptr){
    auto plan = fftw_plan_dft_c2r_1d(size,
                        reinterpret_cast<fftw_complex*>(in_ptr),
                        out_ptr, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}
  
void fft_r2c_2d(size_t width, size_t height, double* in_ptr, xdouble* out_ptr){
    auto plan = fftw_plan_dft_r2c_2d(height, width, in_ptr,
                        reinterpret_cast<fftw_complex*>(out_ptr),
                        FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}
    
void fft_c2r_2d(size_t width, size_t height, xdouble* in_ptr, double* out_ptr){
    auto plan = fftw_plan_dft_c2r_2d(height, width,
                        reinterpret_cast<fftw_complex*>(in_ptr),
                        out_ptr, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

Tensor conv_1d(const Tensor& input, const Tensor& weight){
    auto len = input.shape[1];
    auto x_out = std::make_unique<xdouble[]>(len);
    auto w_out = std::make_unique<xdouble[]>(len);
    
    Tensor result(1, len);
    
    auto w_ptr = weight.data.get();
    auto w_out_ptr = w_out.get();
    fft_r2c_1d(len, w_ptr, w_out_ptr);

    auto x_ptr = input.data.get();
    auto x_out_ptr = x_out.get();
    fft_r2c_1d(len, x_ptr, x_out_ptr);
    
    expand_fftw_arr(len, x_out_ptr);
    expand_fftw_arr(len, w_out_ptr);
    
    auto inv_ptr = std::make_unique<xdouble[]>(len);
    for(size_t i=0; i<len/2; ++i){
        inv_ptr[i] = w_out_ptr[i] * x_out_ptr[i];
    }
    
    auto r_ptr = result.data.get();
    fft_c2r_1d(len, inv_ptr.get(), r_ptr);
    for(size_t i=0; i<len; ++i){
        result.data[i] /= len;
    }
    return result;
}
    
Tensor conv2d(const Tensor& input, const Tensor& weight){
    auto width = input.shape[0];
    auto height = input.shape[1];
    auto size = width * height;
    auto x_out = std::make_unique<xdouble[]>(size);
    auto w_out = std::make_unique<xdouble[]>(size);
    
    Tensor result(width, height);
    
    auto w_ptr = weight.data.get();
    auto w_out_ptr = w_out.get();
    fft_r2c_2d(width, height, w_ptr, w_out_ptr);
    
    auto x_ptr = input.data.get();
    auto x_out_ptr = x_out.get();
    fft_r2c_2d(width, height, x_ptr, x_out_ptr);
    
    expand_fftw_arr(size, x_out_ptr);
    expand_fftw_arr(size, w_out_ptr);
    
    auto inv_ptr = std::make_unique<xdouble[]>(size);
    for(size_t i=0; i<size; ++i){
        inv_ptr[i] = w_out_ptr[i] * x_out_ptr[i];
    }
    
    auto r_ptr = result.data.get();
    fft_c2r_2d(width, height, inv_ptr.get(), r_ptr);
    for(size_t i=0; i<size; i++){
        result.data[i] /= size;
    }
    return result;
}
} // namespace nn
