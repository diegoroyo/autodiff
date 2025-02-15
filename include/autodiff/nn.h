#pragma once

#include <type_traits>

#include "autodiff/types.h"
#include "autodiff/value.h"

namespace ad {
namespace nn {

template <size_t N, typename T,
          typename = std::enable_if_t<!ad::detail::is_value_v<T>>>
auto positional_encoding(_ValueWrapper<T>& v) {
    if constexpr (N == 0) {
        return v;
    }

    // input size
    constexpr size_t IS = ad::detail::is_vec_v<T> ? T::size : 1;
    // output size
    constexpr size_t OS = (2 * N) * IS;
    common::Tensor<float, OS> scales = 0;
    common::Tensor<float, OS> offsets = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 2 * i * IS; j < 2 * i * IS + IS; ++j) {
            scales(j) = std::pow(2, i);
            scales(j + IS) = std::pow(2, i);
            offsets(j) = 0;
            offsets(j + IS) = M_PI_2;
        }
    }

    return ad::sin(ad::expand<2 * N>(v) * scales + offsets);
}

template <size_t H, size_t W, size_t C_IN, size_t C_OUT, size_t K>
auto conv_2d(Tensor<C_IN, H, W>& input, Tensor<C_OUT, C_IN, K, K>& kernel,
             Tensor<C_OUT>& bias);

template <size_t H, size_t W, size_t C_IN>
auto avg_pool_2d(Tensor<C_IN, H, W>& input, size_t kernel_size);

};  // namespace nn
};  // namespace ad

#include "nn.tpp"