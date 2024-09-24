#pragma once

#include <type_traits>

#include "libcpp-common/geometry.h"

namespace ad {
namespace detail {

template <typename T, typename = void>
struct is_value : std::false_type {};
template <typename T>
struct is_value<T, std::void_t<typename T::_ad_value_type>> : std::true_type {};
template <typename T>
inline constexpr bool is_value_v = is_value<T>::value;

template <typename T>
struct is_vec : std::false_type {};
template <typename U, unsigned int N>
struct is_vec<common::Vec<U, N>> : std::true_type {};
template <typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

template <typename T>
struct is_mat : std::false_type {};
template <typename U, unsigned int N>
struct is_mat<common::Mat<U, N>> : std::true_type {};
template <typename U, unsigned int N, unsigned int M>
struct is_mat<common::Mat<U, N, M>> : std::true_type {};
template <typename T>
inline constexpr bool is_mat_v = is_mat<T>::value;

};  // namespace detail
};  // namespace ad