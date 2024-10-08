#include <iostream>

#include "autodiff/autodiff.h"

int main() {
    ad::Value x(-3.14);

    ad::Value y = ad::relu(-x * 3 + 2);
    y.backward();

    std::cout << y << std::endl;
    std::cout << x.grad() << std::endl;
}