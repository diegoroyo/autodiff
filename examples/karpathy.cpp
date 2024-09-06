#include "autodiff/autodiff.h"

int main() {
    ad::Value x(-3);

    auto y = ad::relu(-x * 3 + 2);
    y.backward();

    std::cout << x.grad() << std::endl;
}