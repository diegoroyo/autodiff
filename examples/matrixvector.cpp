#include <array>

#include "autodiff/autodiff.h"

int main() {
    std::array<float, 3> raw_A = {1, 2, 3};
    std::array<float, 3> raw_B = {1, 2, 3};

    ad::Vector<3> v1(raw_A);
    auto y = 2 * v1;
    y.backward();

    std::cout << y << std::endl;
    std::cout << v1.grad() << std::endl;
}