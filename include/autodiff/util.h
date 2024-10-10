#pragma once

#include <iostream>

#include "autodiff/types.h"

#define AD_ENSURE_REQUIRES_GRAD(x)                                       \
    if (!x.m_requires_grad) {                                            \
        std::cerr << "Tried calling backward on a node without gradient" \
                  << std::endl;                                          \
        return;                                                          \
    }

namespace ad {

class ADException : public std::exception {
   private:
    std::string m_message;

   public:
    ADException(const std::string& message) : m_message(message) {}

    virtual const char* what() const noexcept override {
        return m_message.c_str();
    }
};

namespace detail {

template <typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
T sum(const T& v) {
    return v;
}
template <typename T, typename = std::enable_if_t<is_vec_v<T> || is_mat_v<T>>>
typename T::type sum(const T& v) {
    return v.sum();
}

template <typename T, typename R>
auto sum_if_scalar(const R& v) {
    if constexpr (std::is_scalar_v<T>) {
        return sum(v);
    } else {
        return v;
    }
}

template <typename B, typename E,
          typename = std::enable_if_t<std::is_scalar_v<E>>>
B pow(const B& base, const E& exponent) {
    if constexpr (is_vec_v<B> || is_mat_v<B>) {
        return base.pow(exponent);
    } else {
        return std::pow(base, exponent);
    }
}

template <typename T>
T ewise_mult(const T& lhs, const T& rhs) {
    if constexpr (is_vec_v<T> || is_mat_v<T>) {
        return lhs.ewise_mult(rhs);
    } else {
        return lhs * rhs;
    }
}

template <typename T>
T relu_helper(const T& cond, const T& v) {
    if constexpr (is_vec_v<T>) {
        return v.map(
            [&cond](auto& e, size_t i) { return cond[i] > 0 ? e : 0; });
    } else if constexpr (is_mat_v<T>) {
        return v.map([&cond](auto& e, size_t i, size_t j) {
            return cond(i, j) > 0 ? e : 0;
        });
    } else {
        return cond > 0 ? v : 0;
    }
}

template <typename T>
T sigmoid(const T& v) {
    static auto sigmoid_impl = [](float x) { return 1 / (1 + std::exp(-x)); };
    if constexpr (is_vec_v<T>) {
        return v.map([](auto& e, size_t i) { return sigmoid_impl(e); });
    } else if constexpr (is_mat_v<T>) {
        return v.map(
            [](auto& e, size_t i, size_t j) { return sigmoid_impl(e); });
    } else {
        return sigmoid_impl(v);
    }
}

template <typename T>
T sin(const T& v) {
    if constexpr (is_vec_v<T>) {
        return v.map([](auto& e, size_t i) { return std::sin(e); });
    } else if constexpr (is_mat_v<T>) {
        return v.map([](auto& e, size_t i, size_t j) { return std::sin(e); });
    } else {
        return std::sin(v);
    }
}

template <typename T>
T cos(const T& v) {
    if constexpr (is_vec_v<T>) {
        return v.map([](auto& e, size_t i) { return std::cos(e); });
    } else if constexpr (is_mat_v<T>) {
        return v.map([](auto& e, size_t i, size_t j) { return std::cos(e); });
    } else {
        return std::cos(v);
    }
}

template <typename R, typename P, typename B>
static R compute_grad_mult(const P& parent_grad, const B& brother_value) {
    R result;
    // only special cases are matrix * vector and vector * matrix
    // the rest are element-wise operations or just scalar * something
    if constexpr (is_vec_v<R> && is_mat_v<B>) {
        result = brother_value.transpose() * parent_grad;
    } else if constexpr (is_mat_v<R> && is_vec_v<B>) {
        for (unsigned int i = 0; i < R::rows; ++i)
            for (unsigned int j = 0; j < R::cols; ++j)
                result(i, j) = parent_grad[i] * brother_value[j];
    } else {
        result = detail::sum_if_scalar<R>(brother_value * parent_grad);
    }
    return result;
}

};  // namespace detail
};  // namespace ad