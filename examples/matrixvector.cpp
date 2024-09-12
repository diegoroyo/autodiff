#include <array>

#include "autodiff/autodiff.h"

int main() {
    std::array<float, 3> raw_1 = {1, 2, 3};
    ad::Matrix<3, 3> mat1(common::Mat3f::identity());
    ad::Vector<3> v1(raw_1);
    ad::Vector<3> v2({2, 4, 6});
    auto y = (mat1 * v2) + 2;
    auto s = ad::sum(y);
    s.backward();

    std::cout << s << std::endl;
    std::cout << mat1.grad() << std::endl;
}