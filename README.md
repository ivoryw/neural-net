# nn
A framework for building neural network graphs

## Usage
```C++
class Network : public nn::nNet{
public:
    Model() : nn::Net(), fc1(10, 5, this), fc2(5, 1, this){}
    
    nn::fc fc1, fc2;
    audodiff::var forward(autodiff::var& x){
        auto y = fc1(x);
        auto z = fc2(y);
        return z;
    }
};

int main(){
	autodiff::var x(nn::Tensor(...));
	nn::Tensor targets = {...};
	Network net;
	
	auto output = net.forward(x);
	loss.softmax(output, targets);
	net.backwardProp(loss);
}
```
