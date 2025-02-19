#include <array>
#include <random>

#include "autodiff/autodiff.h"

// ad-hoc class for this example
class LeNet {
   private:
    std::default_random_engine m_generator;
    std::normal_distribution<float> m_normal_distribution;
    template <size_t... Shape>
    void fill_tensor(ad::Tensor<Shape...>& t) {
        t.value() = t.value().map([this](auto& e, size_t i) {
            return m_normal_distribution(m_generator);
        });
    }

   public:
    LeNet()
        : m_generator(),
          m_normal_distribution(0.0, 0.1),
          conv1_kernel(0),
          conv1_bias(0),
          conv2_kernel(0),
          conv2_bias(0),
          w1(0),
          b1(0),
          w2(0),
          b2(0),
          w3(0),
          b3(0) {
        fill_tensor(conv1_kernel);
        fill_tensor(conv1_bias);
        fill_tensor(conv2_kernel);
        fill_tensor(conv2_bias);
        fill_tensor(w1);
        fill_tensor(b1);
        fill_tensor(w2);
        fill_tensor(b2);
        fill_tensor(w3);
        fill_tensor(b3);
    }
    ad::Tensor<6, 1, 5, 5> conv1_kernel;
    ad::Tensor<6> conv1_bias;
    ad::Tensor<16, 6, 5, 5> conv2_kernel;
    ad::Tensor<16> conv2_bias;
    ad::Matrix<120, 16 * 7 * 7> w1;
    ad::Vector<120> b1;
    ad::Matrix<84, 120> w2;
    ad::Vector<84> b2;
    ad::Matrix<10, 84> w3;
    ad::Vector<10> b3;

    ad::Vector<10> forward(common::Tensor<float, 1, 28, 28>& image) {
        auto l1 = ad::nn::conv_2d(image, conv1_kernel, conv1_bias);
        auto r1 = ad::relu(l1);
        ad::Tensor<6, 14, 14> o1 = ad::nn::avg_pool_2d<2>(r1);
        auto l2 = ad::nn::conv_2d(o1, conv2_kernel, conv2_bias);
        auto r2 = ad::relu(l2);
        ad::Tensor<16, 7, 7> o2 = ad::nn::avg_pool_2d<2>(r2);

        auto o2_flat = ad::flatten(o2);
        auto l3 = ad::relu(w1 * o2_flat + b1);
        auto l4 = ad::relu(w2 * l3 + b2);
        auto output = w3 * l4 + b3;

        return output;
    }

    void update(const float lr) {
        conv1_kernel.update(lr);
        conv1_bias.update(lr);
        conv2_kernel.update(lr);
        conv2_bias.update(lr);
        w1.update(lr);
        b1.update(lr);
        w2.update(lr);
        b2.update(lr);
        w3.update(lr);
        b3.update(lr);
    }
};

template <size_t x>
void print_sample(const common::Tensor<float, x, 28, 28>& image) {
    for (size_t i = 0; i < 28; ++i) {
        for (size_t j = 0; j < 28; ++j) {
            std::cout << (image(0, i, j) > 0.5f ? "X" : " ");
        }
        std::cout << std::endl;
    }
}

int main() {
    using TrainDataType1u = common::Tensor<uint8_t, 60000, 28, 28>;
    using TrainLabelsType1u = common::Tensor<uint8_t, 60000>;
    using TestDataType1u = common::Tensor<uint8_t, 10000, 28, 28>;
    using TestLabelsType1u = common::Tensor<uint8_t, 10000>;

    auto x_train_raw = TrainDataType1u::load(
        "/media/pleiades/cpp/autodiff/examples/mnist/x_train.npy");
    auto x_train = x_train_raw.cast_to<float>() / 255.0f;
    auto y_train = TrainLabelsType1u::load(
        "/media/pleiades/cpp/autodiff/examples/mnist/y_train.npy");

    auto x_test_raw = TestDataType1u::load(
        "/media/pleiades/cpp/autodiff/examples/mnist/x_test.npy");
    auto x_test = x_test_raw.cast_to<float>() / 255.0f;
    auto y_test = TestLabelsType1u::load(
        "/media/pleiades/cpp/autodiff/examples/mnist/y_test.npy");

    srand(0);
    auto random_engine = std::mt19937(std::random_device{}());
    std::cout << std::setprecision(2) << std::fixed;
    LeNet lenet;
    float lr = 0.001f;
    bool print_while_training = false;
    size_t epochs = 10;

    size_t train_samples = x_train.shape[0];
    size_t test_samples = x_test.shape[0];
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        size_t hits_train = 0;
        std::vector<size_t> random_samples(train_samples);
        std::iota(random_samples.begin(), random_samples.end(), 0);
        std::shuffle(random_samples.begin(), random_samples.end(),
                     random_engine);
        for (size_t i = 0; i < train_samples; ++i) {
            auto image = common::Tensor<float, 1, 28, 28>::zeros();
            std::memcpy(&image.at(0), &x_train(i, 0, 0),
                        28 * 28 * sizeof(float));

            auto label = common::Tensor<float, 10>::zeros();
            label(y_train(i)) = 1.0f;

            auto logits = lenet.forward(image);
            size_t max_pred = 0;
            for (size_t i = 1; i < 10; ++i) {
                if (logits.value()(i) > logits.value()(max_pred)) {
                    max_pred = i;
                }
            }
            if (max_pred == y_train(i)) {
                hits_train++;
            }

            auto loss = ad::nn::cross_entropy(logits, label);

            if (i % 100 == 0) {
                float pct = static_cast<float>(i) * 100 / train_samples;
                float acc = static_cast<float>(hits_train) * 100 / (i + 1);
                std::cout << "Epoch " << epoch << " test ... " << pct
                          << "% (acc " << acc << "%)         \r" << std::flush;
            }

            if (print_while_training && i % 100 == 0) {
                std::cout << "Sample " << i << std::endl;
                print_sample(image);
                size_t max_pred = 0;
                size_t max_gt = 0;
                for (size_t i = 1; i < 10; ++i) {
                    if (logits.value()(i) > logits.value()(max_pred)) {
                        max_pred = i;
                    }
                    if (label(i) > label(max_gt)) {
                        max_gt = i;
                    }
                }
                std::cout << "Prediction: [" << max_pred << "]"
                          << ad::nn::softmax(logits).value() << std::endl;
                std::cout << "Label: [" << max_gt << "]" << label << std::endl;
                std::cout << "Loss: " << loss.value() << std::endl;
            }

            loss.backward();
            lenet.update(lr);
        }

        size_t hits_test = 0;
        for (size_t i = 0; i < test_samples; ++i) {
            auto image = common::Tensor<float, 1, 28, 28>::zeros();
            std::memcpy(&image.at(0), &x_test(i, 0, 0),
                        28 * 28 * sizeof(float));

            auto label = common::Tensor<float, 10>::zeros();
            label(y_test(i)) = 1.0f;

            auto prediction = lenet.forward(image);
            size_t max_pred = 0;
            for (size_t i = 1; i < 10; ++i) {
                if (prediction.value()(i) > prediction.value()(max_pred)) {
                    max_pred = i;
                }
            }
            if (max_pred == y_test(i)) {
                hits_test++;
            }

            if (i % 100 == 0) {
                float pct = static_cast<float>(i) * 100 / test_samples;
                float acc = static_cast<float>(hits_test) * 100 / (i + 1);
                std::cout << "Epoch " << epoch << " test ... " << pct
                          << "% (acc " << acc << "%)         \r" << std::flush;
            }
        }

        float train_accuracy =
            static_cast<float>(hits_train) * 100 / train_samples;
        float test_accuracy =
            static_cast<float>(hits_test) * 100 / test_samples;
        std::cout << "Epoch " << epoch << std::endl;
        std::cout << " - Train accuracy: " << train_accuracy << std::endl;
        std::cout << " - Test accuracy: " << test_accuracy << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
