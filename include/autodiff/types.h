#pragma once

#include <type_traits>

#include "libcpp-common/geometry.h"
#include "libcpp-common/tensor.h"

namespace ad {
namespace detail {

template <typename T>
struct is_value : std::is_same<float, T> {};
template <typename T>
inline constexpr bool is_value_v = is_value<T>::value;

template <typename T>
struct is_vec : std::false_type {};
template <typename U, unsigned int N>
struct is_vec<common::Tensor<U, N>> : std::true_type {};
template <typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

template <typename T>
struct is_mat : std::false_type {};
template <typename U, unsigned int N>
struct is_mat<common::Tensor<U, N>> : std::true_type {};
template <typename U, unsigned int N, unsigned int M>
struct is_mat<common::Tensor<U, N, M>> : std::true_type {};
template <typename T>
inline constexpr bool is_mat_v = is_mat<T>::value;

template <typename T>
struct is_tensor : std::false_type {};
template <typename U, size_t... Shape>
struct is_tensor<common::Tensor<U, Shape...>> : std::true_type {};
template <typename T>
inline constexpr bool is_tensor_v = is_tensor<T>::value;

};  // namespace detail
};  // namespace ad