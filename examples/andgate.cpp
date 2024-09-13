#include <array>

#include "autodiff/autodiff.h"

int main() {
    ad::Matrix<1, 2> w({2, 2});
    ad::Value b(0);

    std::vector<std::array<float, 3>> samples = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 1},
    };

    float lr = 0.1f;

    for (int i = 0; i < 20; ++i) {
        for (const auto& sample : samples) {
            auto [x0, x1, y] = sample;
            common::Vec2f x(x0, x1);

            auto y_est = ad::relu(w * x + b);
            auto loss = ad::pow(y_est - y, 2);
            loss.backward();
            w.update(lr);
            b.update(lr);
        }
    }

    std::cout << "w: " << w << "b: " << b << std::endl;

    for (const auto& sample : samples) {
        auto [x0, x1, y] = sample;
        common::Vec2f x(x0, x1);

        auto y_est = ad::relu(w * x + b);
        std::cout << "x " << x << " y " << y << " y_est " << y_est.value()
                  << std::endl;
    }
}