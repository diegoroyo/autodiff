#include <array>

#include "autodiff/autodiff.h"

int main() {
    ad::Matrix<1, 2> w({1, 1});
    ad::Value b(0);

    std::vector<std::array<float, 3>> samples = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 1},
    };

    float lr = 0.5f;

    for (int i = 0; i < 100; ++i) {
        for (const auto& sample : samples) {
            auto [x0, x1, y] = sample;
            common::Vec2f x(x0, x1);

            auto y_est = ad::relu(w * x + b);
            auto loss = y_est - y;
            loss.backward();

            w.update(lr);
            b.update(lr);
        }
    }
    std::cout << w << " " << b << std::endl;
}