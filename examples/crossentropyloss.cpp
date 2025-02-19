#include <array>

#include "autodiff/autodiff.h"

int main() {
    constexpr size_t N = 3;
    ad::Vector<N> predictions({10, 5, 2});

    auto softmax = ad::nn::softmax(predictions);

    size_t correct_id = 0;
    auto labels = common::Tensor<float, N>::zeros();
    labels(correct_id) = 1.0f;

    auto loss = ad::nn::cross_entropy(softmax, labels);

    std::cout << "Softmax prediction: " << softmax.value() << std::endl;

    loss.backward();
    auto correct = softmax.value();
    correct(correct_id) -= 1.0f;

    std::cout << "AD grad: " << predictions.grad() << std::endl;
    std::cout << "Should be: " << correct << std::endl;
}