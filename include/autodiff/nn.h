#pragma once

#include <type_traits>

#include "autodiff/types.h"
#include "autodiff/value.h"

namespace ad {
namespace nn {

template <unsigned int N, typename T,
          typename = std::enable_if_t<!ad::detail::is_value_v<T>>>
auto positional_encoding(_ValueWrapper<T>& v) {
    if constexpr (N == 0) {
        return v;
    }

    constexpr unsigned int IS = ad::detail::is_vec_v<T> ? T::size : 1;

    constexpr unsigned int OS = (2 * N) * IS;
    common::Tensor<float, OS> scales = 0;
    common::Tensor<float, OS> offsets = 0;
    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 2 * i * IS; j < 2 * i * IS + IS; ++j) {
            scales[j] = std::pow(2, i);
            scales[j + IS] = std::pow(2, i);
            offsets[j] = 0;
            offsets[j + IS] = M_PI_2;
        }
    }

    return ad::sin(ad::expand<2 * N>(v) * scales + offsets);
}

// template <unsigned int C_OUT, unsigned int H, unsigned int W, unsigned int
// C_IN> auto conv2d(const Tensor<H, W, C_IN>& input,
//             const Tensor<H, W, C_IN, C_OUT>& kernel,
//             const Tensor<H, W, C_OUT>& bias) {}

};  // namespace nn
};  // namespace ad