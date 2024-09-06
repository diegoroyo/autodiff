#include <array>

#include "autodiff/autodiff.h"

int main() {
    std::array<float, 3> raw_A = {1, 2, 3};
    std::array<float, 3> raw_B = {1, 2, 3};

    ad::Vector<3> v1(raw_A);
    auto y = v1 * 2;
    auto s = ad::sum(y);
    s.backward();

    std::cout << s << std::endl;
    std::cout << v1.grad() << std::endl;
}