#include <cstdlib>

#include "autodiff/value.h"
#include "libcpp-common/tensor.h"

namespace ad {
namespace nn {

namespace detail {

template <typename T, size_t... Shape, typename... Indices>
const T read_with_padding(common::Tensor<T, Shape...>& tensor,
                          Indices... indices) {
    static_assert(sizeof...(indices) == sizeof...(Shape),
                  "Number of indices must match tensor ndim");
    constexpr size_t ndim = sizeof...(Shape);

    std::array<size_t, ndim> dims = {Shape...};
    std::array<size_t, ndim> idxs = {indices...};
    for (size_t i = 0; i < ndim; ++i) {
        if (idxs[i] < 0 || idxs[i] >= dims[i]) {
            return 0;
        }
    }
    return tensor(indices...);
}

template <typename T, size_t H, size_t W, size_t C_IN, size_t C_OUT, size_t K>
auto conv_2d_impl(common::Tensor<T, C_IN, H, W>& input,
                  common::Tensor<T, C_OUT, C_IN, K, K>& kernel,
                  common::Tensor<T, C_OUT>& bias) {
    constexpr size_t H_OUT = H;
    constexpr size_t W_OUT = W;
    constexpr ssize_t dx = (-(ssize_t)K) / 2;
    constexpr ssize_t dy = dx;

    const auto compute_pixel = [&](size_t c_out, size_t y, size_t x) -> T {
        T output = 0;
        for (size_t c_in = 0; c_in < C_IN; ++c_in) {
            for (size_t ky = 0; ky < K; ++ky) {
                for (size_t kx = 0; kx < K; ++kx) {
                    T e = read_with_padding(input, c_in, y + ky + dy,
                                            x + kx + dx);
                    output += e * kernel(c_out, c_in, ky, kx);
                }
            }
        }
        output += bias(c_out);
        return output;
    };

    auto output = common::Tensor<T, C_OUT, H_OUT, W_OUT>::zeros();
    for (size_t y = 0; y < H_OUT; ++y) {
        for (size_t x = 0; x < W_OUT; ++x) {
            for (size_t c_out = 0; c_out < C_OUT; ++c_out) {
                output(c_out, y, x) = compute_pixel(c_out, y, x);
            }
        }
    }

    return output;
}

template <typename KernelType, size_t H, size_t W,
          typename T = typename KernelType::type,
          size_t C_IN = KernelType::template get_dim<1>()>
KernelType conv_2d_kernel_grad(common::Tensor<T, C_IN, H, W>& grad,
                               common::Tensor<T, C_IN, H, W>& value) {
    constexpr size_t C_OUT = KernelType::template get_dim<0>();
    constexpr size_t K = KernelType::template get_dim<2>();

    const auto compute_pixel = [&](size_t c_in, ssize_t dy, size_t dx) -> T {
        T output = 0;
        for (size_t y = 0; y < H; ++y) {
            for (size_t x = 0; x < W; ++x) {
                output += read_with_padding(grad, c_in, y + dy, x + dx) *
                          read_with_padding(value, c_in, y + dy, x + dx);
            }
        }
        return output;
    };

    constexpr ssize_t dx = (-(ssize_t)K) / 2;
    constexpr ssize_t dy = dx;

    auto kernel_grad = KernelType::zeros();
    for (size_t ky = 0; ky < K; ++ky) {
        for (size_t kx = 0; kx < K; ++kx) {
            for (size_t c_in = 0; c_in < C_IN; ++c_in) {
                T output = compute_pixel(c_in, ky + dy, kx + dx);
                for (size_t c_out = 0; c_out < C_OUT; ++c_out) {
                    kernel_grad(c_out, c_in, ky, kx) = output;
                }
            }
        }
    }

    return kernel_grad;
}

template <size_t kernel_size, typename T, size_t H, size_t W, size_t C_IN>
auto avg_pool_2d_impl(common::Tensor<T, C_IN, H, W>& input) {
    // division rounding up
    constexpr size_t H_OUT = H / kernel_size + (H % kernel_size != 0);
    constexpr size_t W_OUT = W / kernel_size + (H % kernel_size != 0);

    const auto compute_pixel = [&](size_t c_in, ssize_t y, size_t x) -> T {
        T sum = 0;
        for (size_t ky = 0; ky < kernel_size; ++ky) {
            for (size_t kx = 0; kx < kernel_size; ++kx) {
                sum += read_with_padding(input, c_in,  //
                                         y * kernel_size + ky,
                                         x * kernel_size + kx);
            }
        }
        return sum / (kernel_size * kernel_size);
    };

    auto output = common::Tensor<T, C_IN, H_OUT, W_OUT>::zeros();
    for (size_t c_in = 0; c_in < C_IN; ++c_in) {
        for (size_t y = 0; y < H_OUT; ++y) {
            for (size_t x = 0; x < W_OUT; ++x) {
                output(c_in, y, x) = compute_pixel(c_in, y, x);
            }
        }
    }

    return output;
}

};  // namespace detail

template <size_t H, size_t W, size_t C_IN, size_t C_OUT, size_t K>
auto conv_2d(Tensor<C_IN, H, W>& input, Tensor<C_OUT, C_IN, K, K>& kernel,
             Tensor<C_OUT>& bias) {
    using InputType = typename Tensor<C_IN, H, W>::type;
    using KernelType = typename Tensor<C_OUT, C_IN, K, K>::type;
    using BiasType = typename Tensor<C_OUT>::type;
    using OutputType = typename Tensor<C_OUT, H, W>::type;
    static auto backward_f = [](_ValueData<OutputType>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<InputType>& input = v.template get_child<InputType>(0);
        _ValueData<KernelType>& kernel = v.template get_child<KernelType>(1);
        _ValueData<BiasType>& bias = v.template get_child<BiasType>(2);
        if (input.m_requires_grad) {
            auto zeros = BiasType::zeros();
            input.m_grad =
                detail::conv_2d_impl(v.m_grad, kernel.value(), zeros);
            input.backward();
        }
        if (kernel.m_requires_grad) {
            kernel.m_grad = detail::conv_2d_kernel_grad<KernelType>(
                v.m_grad, input.value());
            kernel.backward();
        }
        if (bias.m_requires_grad) {
            bias.m_grad = ad::detail::sum(v.m_grad);
            bias.backward();
        }
    };
    static auto to_string =
        [](std::ostream& o, const _ValueData<OutputType>& v) -> std::ostream& {
        _ValueData<InputType>& input = v.template get_child<InputType>(0);
        _ValueData<KernelType>& kernel = v.template get_child<KernelType>(1);
        _ValueData<BiasType>& bias = v.template get_child<BiasType>(2);
        o << "conv2d(input=" << input << ", kernel=" << kernel
          << ", bias=" << bias << ")";
        return o;
    };
    _ValueWrapper<OutputType> result(
        detail::conv_2d_impl(input.value(), kernel.value(), bias.value()),
        backward_f, to_string, "conv2d",
        {AD_CHILD(input), AD_CHILD(kernel), AD_CHILD(bias)});
    input.set_parent(result);
    kernel.set_parent(result);
    bias.set_parent(result);
    return result;
}
template <size_t H, size_t W, size_t C_IN, size_t C_OUT, size_t K>
auto conv_2d(Tensor<C_IN, H, W>&& input, Tensor<C_OUT, C_IN, K, K>& kernel,
             Tensor<C_OUT>& bias) {
    return conv_2d(input, kernel, bias);
}
template <size_t C_IN, size_t C_OUT, size_t K, typename T,
          typename = std::enable_if_t<ad::detail::is_tensor_v<T>>>
auto conv_2d(T& input, Tensor<C_OUT, C_IN, K, K>& kernel, Tensor<C_OUT>& bias) {
    return conv_2d(AD_MAKE_TEMP(input, T), kernel, bias);
}
template <size_t C_IN, size_t C_OUT, size_t K, typename T,
          typename = std::enable_if_t<ad::detail::is_tensor_v<T>>>
auto conv_2d(T&& input, Tensor<C_OUT, C_IN, K, K>& kernel,
             Tensor<C_OUT>& bias) {
    return conv_2d(AD_MAKE_TEMP(input, T), kernel, bias);
}

template <size_t kernel_size, size_t C_IN, size_t H, size_t W>
auto avg_pool_2d(Tensor<C_IN, H, W>& input) {
    using InputType = typename Tensor<C_IN, H, W>::type;
    // division rounding up
    constexpr size_t H_OUT = H / kernel_size + (H % kernel_size != 0);
    constexpr size_t W_OUT = W / kernel_size + (H % kernel_size != 0);
    using OutputType = typename Tensor<C_IN, H_OUT, W_OUT>::type;
    static auto backward_f = [](_ValueData<OutputType>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<InputType>& input = v.template get_child<InputType>(0);
        if (input.m_requires_grad) {
            using T = typename InputType::type;
            const auto set_pixel = [&](size_t c_in, size_t y, size_t x,
                                       size_t ky, size_t kx) {
                T grad = detail::read_with_padding(input.value(), c_in, y, x);
                size_t py = y * kernel_size + ky;
                size_t px = x * kernel_size + kx;
                if (py >= H || px >= W) return;
                input.m_grad(c_in, py, px) = grad / (kernel_size * kernel_size);
            };
            for (size_t c_in = 0; c_in < C_IN; ++c_in) {
                for (size_t y = 0; y < H_OUT; ++y) {
                    for (size_t x = 0; x < W_OUT; ++x) {
                        for (size_t ky = 0; ky < kernel_size; ++ky) {
                            for (size_t kx = 0; kx < kernel_size; ++kx) {
                                set_pixel(c_in, y, x, ky, kx);
                            }
                        }
                    }
                }
            }
            std::cout << input.m_grad << std::endl;
            input.backward();
        }
    };
    static auto to_string =
        [](std::ostream& o, const _ValueData<OutputType>& v) -> std::ostream& {
        _ValueData<InputType>& input = v.template get_child<InputType>(0);
        o << "avgpool2d(" << input << ", size=" << kernel_size << ")";
        return o;
    };
    _ValueWrapper<OutputType> result(
        ad::nn::detail::avg_pool_2d_impl<kernel_size>(input.value()),
        backward_f, to_string, "avgpool2d", {AD_CHILD(input)});
    input.set_parent(result);
    return result;
}
template <size_t kernel_size, size_t C_IN, size_t H, size_t W>
auto avg_pool_2d(Tensor<C_IN, H, W>&& input) {
    return avg_pool_2d<kernel_size>(input);
}
template <size_t kernel_size, typename T, size_t C_IN, size_t H, size_t W,
          typename = std::enable_if_t<ad::detail::is_tensor_v<T>>>
auto avg_pool_2d(T& input) {
    return avg_pool_2d<kernel_size>(AD_MAKE_TEMP(input, T));
}
template <size_t kernel_size, typename T, size_t C_IN, size_t H, size_t W,
          typename = std::enable_if_t<ad::detail::is_tensor_v<T>>>
auto avg_pool_2d(T&& input) {
    return avg_pool_2d<kernel_size>(AD_MAKE_TEMP(input, T));
}

};  // namespace nn
};  // namespace ad