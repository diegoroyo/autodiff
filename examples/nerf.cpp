#include <array>
#include <random>

#include "autodiff/autodiff.h"
#include "autodiff/nn.h"
#include "libcpp-common/bitmap.h"

// ad-hoc class for this example
class NeRF {
   private:
    std::default_random_engine m_generator;
    std::normal_distribution<float> m_normal_distribution;
    template <unsigned int N, unsigned int M>
    void fill_matrix(ad::Matrix<N, M>& m) {
        for (unsigned int i = 0; i < N; ++i)
            for (unsigned int j = 0; j < M; ++j)
                m.value()(i, j) = m_normal_distribution(m_generator);
    }
    template <unsigned int N>
    void fill_vector(ad::Vector<N>& m) {
        for (unsigned int i = 0; i < N; ++i)
            m.value()[i] = m_normal_distribution(m_generator);
    }

   public:
    NeRF()
        : m_generator(),
          m_normal_distribution(0.0, 0.1),
          w1(0),
          w2(0),
          w3(0),
          w4(0),
          b1(0),
          b2(0),
          b3(0),
          b4(0) {
        fill_matrix(w1);
        fill_matrix(w2);
        fill_matrix(w3);
        fill_matrix(w4);
        fill_vector(b1);
        fill_vector(b2);
        fill_vector(b3);
        fill_vector(b4);
    }
    ad::Matrix<128, 32> w1;
    ad::Matrix<128, 128> w2, w3;
    ad::Matrix<3, 128> w4;
    ad::Vector<128> b1, b2, b3;
    ad::Vector<3> b4;

    ad::Vector<3>& forward(ad::Vector<2>& xy) {
        ad::Vector<32>& input = ad::nn::positional_encoding<8>(xy);

        ad::Vector<128>& l1 = ad::relu(w1 * input + b1);
        ad::Vector<128>& l2 = ad::relu(w2 * l1 + b2);
        ad::Vector<128>& l3 = ad::relu(w3 * l2 + b3);
        ad::Vector<3>& output = ad::sigmoid(w4 * l3 + b4);

        return output;
    }

    void update(const float lr) {
        w1.update(lr);
        w2.update(lr);
        w3.update(lr);
        w4.update(lr);
        b1.update(lr);
        b2.update(lr);
        b3.update(lr);
        b4.update(lr);
    }
};

void save_image(NeRF& nerf, unsigned int width, unsigned int height,
                size_t step) {
    common::Bitmap3f y_est(width, height, 0);

    for (unsigned int px = 0; px < width; ++px) {
        for (unsigned int py = 0; py < height; ++py) {
            ad::Vector<2> xy({(float)px / width, (float)py / height});
            common::Color3f value = nerf.forward(xy).value();
            y_est(px, py) = value;
        }
    }

    common::save_bitmap(
        "/home/diego/cpp/autodiff/examples/gif/sunflower_nerf_est" +
            std::to_string(step) + ".ppm",
        y_est);
}

int main() {
    common::Bitmap3u image = common::load_bitmap<common::Color3u>(
        "/home/diego/cpp/autodiff/examples/images/sunflower.ppm");
    common::Bitmap3f y = image.map<common::Color3f>(
        [](const common::Color3u& in) { return in.cast_to<float>() / 255.0f; });

    auto [width, height] = y.size();

    srand(0);
    NeRF nerf;

    float lr = 0.15f;
    size_t steps = 200001;

    for (size_t step = 0; step < steps; ++step) {
        std::cout << "." << std::flush;
        unsigned int px = rand() % width;
        unsigned int py = rand() % height;

        ad::Vector<2> xy({(float)px / width, (float)py / height});
        common::Vec3f y_i = y(px, py);

        auto& y_est = nerf.forward(xy);
        auto& loss = ad::pow(y_est - y_i, 2);
        loss.backward();
        nerf.update(lr);

        if (step < 2500 && step % 250 == 0) {
            save_image(nerf, width, height, step);
        } else if (step < 10000 && step % 1000 == 0) {
            save_image(nerf, width, height, step);
        } else if (step < 50000 && step % 5000 == 0) {
            save_image(nerf, width, height, step);
        } else if (step % 10000 == 0) {
            save_image(nerf, width, height, step);
        }
    }
}