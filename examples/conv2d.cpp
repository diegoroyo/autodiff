#include "autodiff/autodiff.h"
#include "libcpp-common/tensor.h"

int main() {
    ad::Tensor<1, 3, 3> image({{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    }});
    // image.set_requires_grad(false);
    ad::Tensor<1, 1, 3, 3> kernel({{{
        {3, 3, 3},
        {3, 3, 3},
        {3, 3, 3},
    }}});
    ad::Tensor<1> bias({0});

    auto output = ad::nn::conv_2d(image + 1 - 1, kernel, bias);
    auto avgpool = ad::nn::avg_pool_2d<2>(output);

    std::cout << avgpool << std::endl;
    std::cout << avgpool.value() << std::endl;

    auto l = ad::sum(output);
    l.backward();

    std::cout << image.grad() << std::endl;
    std::cout << kernel.grad() << std::endl;
    std::cout << bias.grad() << std::endl;

    ad::Tensor<1, 7, 7> image2(0);

    auto avgpool2 = ad::nn::avg_pool_2d<2>(image2);
    std::cout << avgpool2 << std::endl;

    auto l2 = ad::sum(avgpool2) * 2;
    l2.backward();

    std::cout << image2.grad() << std::endl;
}

/*

Matrix:
 tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
Kernel:
 tensor([[3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.]])
Result:
 tensor([[ 36.,  63.,  48.],
        [ 81., 135.,  99.],
        [ 72., 117.,  84.]])

Matrix gradient:
 tensor([[12., 18., 12.],
        [18., 27., 18.],
        [12., 18., 12.]])

Kernel gradient:
 tensor([[12., 21., 16.],
        [27., 45., 33.],
        [24., 39., 28.]])

*/