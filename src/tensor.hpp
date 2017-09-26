/**
    Tensor
    A tensor module for linear algebra
 */
#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <array>

namespace nn {

// Forward declarations
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
Tensor conv_1d(const Tensor&, const Tensor&);
Tensor conv2d(const Tensor&, const Tensor&);
double dot(const Tensor& lhs, const Tensor& rhs);

typedef std::array<size_t, 4> Shape;
  
/**
    Tensor
    A tensor object for linear algebra
*/
class Tensor {
    private:
        std::unique_ptr<double[]> data;
    public:
        // Can be initialised using a size, an array of sizes with a constant or another tensor
        Tensor(size_t, size_t=1, size_t=1, size_t=1);
        Tensor(const Shape&);
        Tensor(const Shape&, double);
        Tensor(const Tensor&);
        Tensor();
        
        size_t size;
        Shape shape;
        
        // Random access iterator to internal data pointer
        class Iterator : public std::iterator<std::random_access_iterator_tag, double> {
        private:
            double* data_ptr;
        public:
            Iterator(double* data_ptr_) : data_ptr(data_ptr_) {}
            Iterator& operator++() { ++data_ptr; return *this; }
            Iterator operator++(int) { Iterator tmp(*this); operator++(); return tmp; }
            bool operator==(const Iterator& rhs) { return data_ptr == rhs.data_ptr; }
            bool operator!=(const Iterator& rhs){ return data_ptr != rhs.data_ptr; }
            double& operator*() { return *data_ptr; }
        };
        
        Iterator begin() const { return Iterator(data.get()); }
        Iterator end() const { return Iterator(data.get()+size); }
        
        // Formatted ostream
        friend std::ostream& operator<<(std::ostream&, const Tensor&);
        
        // Element Access
        // Exceptions thrown if invalid elements are accessed
        double &operator()(size_t x, size_t y=0, size_t z=0, size_t t=0);
        double operator()(size_t x, size_t y=0, size_t z=0, size_t t=0) const;
        
        // Sub matrix access
        Tensor row(size_t);
        Tensor col(size_t);
        Tensor slice(size_t);
        Tensor tube(size_t, size_t, size_t, size_t);
        Tensor sub_mat(size_t, size_t, size_t, size_t);

        // Sub matrix assignment
        void set_row(size_t, const Tensor&);
        void set_col(size_t, const Tensor&);
        void set_slice(size_t, const Tensor&);
        void set_tube(size_t, size_t, size_t, size_t, const Tensor&);
        void set_sub_mat(size_t, size_t, size_t, const Tensor&);
        
        void operator=(const Tensor&);
        Tensor operator+(const Tensor&) const;      // Pointwise addition
        Tensor operator-(const Tensor&) const;      // Pointwise subtraction
        Tensor operator*(const Tensor&) const;      // Matrix multiplication
        Tensor operator/(const Tensor&) const;      // Pointwise division
        Tensor operator*(double)const;              // Scalar multiplication
        Tensor operator/(double)const;              // Scalar division
        Tensor operator%(const Tensor&) const;      // Pointwise multiplication
        
        void operator+=(const Tensor&);
        void operator-=(const Tensor&);
        void operator*=(const Tensor&);
        void operator/=(const Tensor&);
        void operator/=(double);
        void operator%=(const Tensor&);
        
        // Transpose
        Tensor t()const;
        // Dot product
        friend double nn::dot(const Tensor& lhs, const Tensor& rhs);
        
        // Constant arithmatic
        friend Tensor operator+(double, const Tensor&);
        friend Tensor operator-(double, const Tensor&);
        friend Tensor operator*(double, const Tensor&);
        friend Tensor operator/(double, const Tensor&);
        friend Tensor operator/(const Tensor&, double);
        
        // Initialisers
        void rand(double=0, double=1);
        void rand_int(int=0, int=10);
        void randn(double=0, double=1);
        void ones();
        void zeros();
        void constant(double);

        // Reductions
        double abs_sum();
        double sum();
        
        // Trig & and arithmatic overloads
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
        friend Tensor conv_1d(const Tensor&, const Tensor&);
        friend Tensor conv2d(const Tensor&, const Tensor&);
    };
} // namespace nn

#endif // TENSOR_H
