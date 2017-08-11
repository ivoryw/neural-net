/***************************************************
 Tensor
 Values are stored as a contiguous array and accessed externally using
 brackets notation (x,y,z,t). Internally the array indecies are used.
 CBLAS is used for as many operations as feasable
 Methods are provided for common usages.
 Some may be split out as standalone functions at a later date.
 **************************************************/
#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <array>

namespace nn{
    class Tensor;
    
    Tensor sin(const Tensor& rhs);
    Tensor cos(const Tensor& rhs);
    Tensor tan(const Tensor& rhs);
    Tensor asin(const Tensor& rhs);
    Tensor acos(const Tensor& rhs);
    Tensor atan(const Tensor& rhs);
    Tensor pow(const Tensor&, double);
    Tensor pow(const Tensor&, const Tensor&);
    Tensor log(const Tensor&);
    Tensor sqrt(const Tensor&);
    Tensor conv1d(const Tensor&, const Tensor&);
    Tensor conv2d(const Tensor&, const Tensor&);
    double dot(const Tensor& lhs, const Tensor& rhs);
    
    typedef std::array<size_t, 4> Shape;
    
    class Tensor{
    private:
        std::unique_ptr<double[]> data;
    public:
        Tensor(size_t, size_t=1, size_t=1, size_t=1);
        Tensor(const Shape&);
        Tensor(const Shape&, double);
        Tensor(const Tensor&);
        Tensor();
        
        size_t size;
        Shape shape;
        
        // Random access iterator to internal data pointer
        class iterator
        : public std::iterator<std::random_access_iterator_tag, double>{
            double* data_ptr;
        public:
            iterator(double* data_ptr_) : data_ptr(data_ptr_){}
            iterator& operator++(){ ++data_ptr; return *this; }
            iterator operator++(int){ iterator tmp(*this); operator++(); return tmp; }
            bool operator==(const iterator& rhs){ return data_ptr == rhs.data_ptr; }
            bool operator!=(const iterator& rhs){ return data_ptr != rhs.data_ptr; }
            double& operator*(){ return *data_ptr; }
        };
        
        iterator begin() const { return iterator(data.get()); }
        iterator end() const { return iterator(data.get()+size); }
        
        /************************************************************
         Operators
         Plan to handle as many ordinary tensor operations through provided
         functions or operators.
         These overloads cover Tensor-Tensor operations
         std::invalid argument is thrown if the operation won't work with
         given tensor dimensions
         ************************************************************/
        
        friend std::ostream& operator<<(std::ostream&, const Tensor&);
        
        // Element Access
        // Exceptions thrown if invalid elements are accessed
        double &operator()(size_t x, size_t y=0, size_t z=0, size_t t=0);
        double operator()(size_t x, size_t y=0, size_t z=0, size_t t=0) const;
        
        void operator=(const Tensor&);
        Tensor operator+(const Tensor&)const;
        Tensor operator-(const Tensor&)const;
        Tensor operator*(const Tensor&)const;
        Tensor operator/(const Tensor&)const;        // Element-wise matrix division
        Tensor operator*(double)const;
        Tensor operator/(double)const;
        Tensor operator%(const Tensor&) const;   // Elementwise multiplication
        
        void operator+=(const Tensor&);
        void operator-=(const Tensor&);
        void operator*=(const Tensor&);
        void operator/=(const Tensor&);
        void operator/=(double);
        void operator%=(const Tensor&);
        
        Tensor t()const;
        friend double nn::dot(const Tensor& lhs, const Tensor& rhs);
        
        friend Tensor operator+(double, const Tensor&);
        friend Tensor operator-(double, const Tensor&);
        friend Tensor operator*(double, const Tensor&);
        friend Tensor operator/(double, const Tensor&);
        friend Tensor operator/(const Tensor&, double);
        
        void rand(double=0, double=1);
        void rand_int(int=0, int=10);
        void randn(double=0, double=1);
        void ones();
        void zeros();
        void constant(double);
        
        friend Tensor sin(const Tensor& rhs);
        friend Tensor cos(const Tensor& rhs);
        friend Tensor tan(const Tensor& rhs);
        friend Tensor asin(const Tensor& rhs);
        friend Tensor acos(const Tensor& rhs);
        friend Tensor atan(const Tensor& rhs);
        friend Tensor pow(const Tensor&, double);
        friend Tensor pow(const Tensor&, const Tensor&);
        friend Tensor log(const Tensor&);
        friend Tensor sqrt(const Tensor&);
        friend Tensor conv1d(const Tensor&, const Tensor&);
        friend Tensor conv2d(const Tensor&, const Tensor&);
    };
} // namespace nn

#endif // TENSOR_H
