#pragma once

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "libcpp-common/geometry.h"

namespace ad {

namespace detail {

template <typename T, typename = void>
struct is_value : std::false_type {};
template <typename T>
struct is_value<T, std::void_t<typename T::_value_type>> : std::true_type {};
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

template <typename T1, typename T2,
          typename = std::enable_if_t<std::is_scalar_v<T2>>>
T1 pow(const T1& base, const T2& exponent) {
    if constexpr (is_vec_v<T1> || is_mat_v<T1>) {
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
        return v > 0 ? v : 0;
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

template <typename TR, typename TP, typename TB>
static TR compute_grad_mult(const TP& parent_grad, const TB& brother_value) {
    TR result;
    // only special cases are matrix * vector and vector * matrix
    // the rest are element-wise operations or just scalar * something
    if constexpr (is_vec_v<TR> && is_mat_v<TB>) {
        result = brother_value.transpose() * parent_grad;
    } else if constexpr (is_mat_v<TR> && is_vec_v<TB>) {
        for (unsigned int i = 0; i < TR::rows; ++i)
            for (unsigned int j = 0; j < TR::cols; ++j)
                result(i, j) = parent_grad[i] * brother_value[j];
    } else {
        result = detail::sum_if_scalar<TR>(brother_value * parent_grad);
    }
    return result;
}

};  // namespace detail

#define AD_ENSURE_REQUIRES_GRAD(x)                                       \
    if (!x->m_requires_grad) {                                           \
        std::cerr << "Tried calling backward on a node without gradient" \
                  << std::endl;                                          \
        return;                                                          \
    }

template <typename T, typename = std::enable_if_t<std::is_scalar_v<T> ||
                                                  detail::is_vec_v<T> ||
                                                  detail::is_mat_v<T>>>
class _Value {
   public:
    using _value_type = T;
    using type = T;

    template <typename U, typename>
    friend class _Value;

   private:
    T m_value;
    T m_grad;
    bool m_has_grad;
    bool m_requires_grad;
    void (*m_backward_f)(_Value<T>*);
    std::ostream& (*m_to_string)(std::ostream&, const _Value<T>*);
    std::string m_op_name;
    // children/parent can be of different T
    void* m_parent;
    std::vector<void*> m_children;

    static std::ostream& default_to_string(std::ostream& o,
                                           const _Value<T>* v) {
        o << v->m_value;
        return o;
    }

   public:
    explicit _Value(
        T value, void (*backward_f)(_Value<T>*) = [](_Value<T>*) {},
        std::ostream& (*to_string)(std::ostream&, const _Value<T>*) =
            &_Value<T>::default_to_string,
        std::string op_name = "Value", std::vector<void*> const& children = {})
        : m_value(value),
          m_grad(1.0f),
          m_has_grad(false),
          m_requires_grad(true),
          m_backward_f(backward_f),
          m_to_string(to_string),
          m_op_name(op_name),
          m_parent(nullptr),
          m_children(children) {}
    // TODO handle parent node in destructor (destroy it)
    // ~_Value() { std::cerr << "Destructor called" << std::endl; }

    static _Value<T>* TempValue(T value) {
        _Value<T>* result = new _Value<T>(value);
        result->m_requires_grad = false;
        return result;
    }

    void backward() {
        m_has_grad = true;
        m_backward_f(this);
    }
    T& value() { return m_value; }
    T& grad() {
        if (!m_has_grad) {
            std::cerr << "grad() called on a node without computed gradient"
                      << std::endl;
        }
        return m_grad;
    }
    void update(float lr) { m_value -= grad() * lr; }

    inline friend std::ostream& operator<<(std::ostream& s, const _Value& v) {
        return v.m_to_string(s, &v);
    }

#define AD_MAKE_TEMP(x, t) *_Value<t>::TempValue(x)
#define AD_TEMPLATE_NON_CLS(op)                                          \
    template <typename T2,                                               \
              typename = std::enable_if_t<!detail::is_value<T>::value && \
                                          !detail::is_value<T2>::value>, \
              typename TR = decltype(std::declval<T>() op std::declval<T2>())>
// there are many ways to call a simple addition operation
// first, lhs and/or rhs can be lvalue or rvalue (that's why the first 3
// methods are here). second, lhs and/or rhs can be from a class that is not
// _Value. In those cases we make a TempValue as seen above. That's the rest
// of the methods of this list.
// the is_same_template are there to prevent _Value<_Value<T>> recursiveness
#define AD_BINARY_PERFECT_FORWARD(op, func)                      \
    AD_TEMPLATE_NON_CLS(op)                                      \
    friend _Value<TR>& func(_Value<T>& lhs, _Value<T2>&& rhs) {  \
        return func(lhs, rhs);                                   \
    }                                                            \
    AD_TEMPLATE_NON_CLS(op)                                      \
    friend _Value<TR>& func(_Value<T>&& lhs, _Value<T2>& rhs) {  \
        return func(lhs, rhs);                                   \
    }                                                            \
    AD_TEMPLATE_NON_CLS(op)                                      \
    friend _Value<TR>& func(_Value<T>&& lhs, _Value<T2>&& rhs) { \
        return func(lhs, rhs);                                   \
    }                                                            \
    AD_TEMPLATE_NON_CLS(op)                                      \
    friend _Value<TR>& func(_Value<T>& lhs, T2 rhs) {            \
        return func(lhs, AD_MAKE_TEMP(rhs, T2));                 \
    }                                                            \
    AD_TEMPLATE_NON_CLS(op)                                      \
    friend _Value<TR>& func(_Value<T>&& lhs, T2 rhs) {           \
        return func(lhs, AD_MAKE_TEMP(rhs, T2));                 \
    }                                                            \
    AD_TEMPLATE_NON_CLS(op)                                      \
    friend _Value<TR>& func(T lhs, _Value<T2>& rhs) {            \
        return func(AD_MAKE_TEMP(lhs, T), rhs);                  \
    }                                                            \
    AD_TEMPLATE_NON_CLS(op)                                      \
    friend _Value<TR>& func(T lhs, _Value<T2>&& rhs) {           \
        return func(AD_MAKE_TEMP(lhs, T), rhs);                  \
    }

#define AD_BINARY_OP(op, lhs_grad, rhs_grad)                                   \
    template <typename T2,                                                     \
              typename TR = decltype(std::declval<T>() op std::declval<T2>())> \
    friend _Value<TR>& operator op(_Value<T>& lhs, _Value<T2>& rhs) {          \
        static auto backward_f = [](_Value<TR>* v) {                           \
            AD_ENSURE_REQUIRES_GRAD(v);                                        \
            _Value<T>* lhs = static_cast<_Value<T>*>(v->m_children[0]);        \
            _Value<T2>* rhs = static_cast<_Value<T2>*>(v->m_children[1]);      \
            if (lhs->m_requires_grad) {                                        \
                lhs->m_grad = lhs_grad;                                        \
                lhs->backward();                                               \
            }                                                                  \
            if (rhs->m_requires_grad) {                                        \
                rhs->m_grad = rhs_grad;                                        \
                rhs->backward();                                               \
            }                                                                  \
        };                                                                     \
        static auto to_string = [](std::ostream& o,                            \
                                   const _Value<TR>* v) -> std::ostream& {     \
            _Value<T>* lhs = static_cast<_Value<T>*>(v->m_children[0]);        \
            _Value<T2>* rhs = static_cast<_Value<T2>*>(v->m_children[1]);      \
            o << *lhs << #op << *rhs;                                          \
            return o;                                                          \
        };                                                                     \
        _Value<TR>* result =                                                   \
            new _Value<TR>(lhs.m_value op rhs.m_value, backward_f, to_string,  \
                           #op, {&lhs, &rhs});                                 \
        lhs.m_parent = result;                                                 \
        rhs.m_parent = result;                                                 \
        return *result;                                                        \
    }                                                                          \
    AD_BINARY_PERFECT_FORWARD(op, operator op)

    AD_BINARY_OP(+, detail::sum_if_scalar<decltype(lhs->m_value)>(v->m_grad),
                 detail::sum_if_scalar<decltype(rhs->m_value)>(v->m_grad));
    AD_BINARY_OP(-, detail::sum_if_scalar<decltype(lhs->m_value)>(v->m_grad),

                 detail::sum_if_scalar<decltype(rhs->m_value)>(v->m_grad) *
                     -1.0f);
    AD_BINARY_OP(*, detail::compute_grad_mult<T>(v->m_grad, rhs->m_value),
                 detail::compute_grad_mult<T2>(v->m_grad, lhs->m_value));
    AD_BINARY_OP(/,
                 detail::sum_if_scalar<decltype(lhs->m_value)>(v->m_grad /
                                                               rhs->m_value),
                 detail::sum_if_scalar<decltype(rhs->m_value)>(
                     v->m_grad * lhs->m_value / (rhs->m_value * rhs->m_value)));

#define AD_UNARY_OP(op, value, grad)                                       \
    template <typename TR = decltype(op std::declval<T>())>                \
    friend _Value<TR>& operator op(_Value<T>& obj) {                       \
        static auto backward_f = [](_Value<TR>* v) {                       \
            AD_ENSURE_REQUIRES_GRAD(v);                                    \
            _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);    \
            if (obj->m_requires_grad) {                                    \
                obj->m_grad = grad;                                        \
                obj->backward();                                           \
            }                                                              \
        };                                                                 \
        static auto to_string = [](std::ostream& o,                        \
                                   const _Value<TR>* v) -> std::ostream& { \
            _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);    \
            o << #op << *obj;                                              \
            return o;                                                      \
        };                                                                 \
        _Value<TR>* result =                                               \
            new _Value<TR>(value, backward_f, to_string, #op, {&obj});     \
        obj.m_parent = result;                                             \
        return *result;                                                    \
    }                                                                      \
    template <typename TR = decltype(op std::declval<T>())>                \
    friend _Value<TR>& operator op(_Value<T>&& obj) {                      \
        return operator op(obj);                                           \
    }

    AD_UNARY_OP(-, -obj.m_value, v->m_grad * -1);

    template <typename T1, typename T2, typename>
    friend _Value<T1>& pow(_Value<T1>& base, _Value<T2>& exponent);
    template <typename T1>
    friend _Value<T1>& relu(_Value<T1>& obj);
    template <typename T1>
    friend _Value<T1>& sigmoid(_Value<T1>& obj);
    template <typename T1>
    friend _Value<float>& sum(_Value<T1>& obj);
    template <typename T1>
    friend _Value<T1>& sin(_Value<T1>& obj);
    template <typename T1>
    friend _Value<T1>& cos(_Value<T1>& obj);
    template <unsigned int N>
    friend _Value<common::Vec<float, N>>& expand(_Value<float>& obj);
    template <unsigned int N, unsigned int S>
    friend _Value<common::Vec<float, S * N>>& expand(
        _Value<common::Vec<float, S>>& obj);
};

using Value = _Value<float>;

/// Math functions ///

#define AD_POW_TEMPLATE                                                   \
    template <typename T1, typename T2,                                   \
              typename = std::enable_if_t<!detail::is_value<T1>::value && \
                                          std::is_scalar_v<T2>>>
AD_POW_TEMPLATE
_Value<T1>& pow(_Value<T1>& base, _Value<T2>& exponent) {
    static auto backward_f = [](_Value<T1>* v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _Value<T1>* base = static_cast<_Value<T1>*>(v->m_children[0]);
        _Value<T2>* exponent = static_cast<_Value<T2>*>(v->m_children[1]);
        if (base->m_requires_grad) {
            // d/dx x^n = n*x^(n-1)
            base->m_grad = ad::detail::ewise_mult(
                v->m_grad * exponent->m_value,
                ad::detail::pow(base->m_value, exponent->m_value - 1.0f));
            base->backward();
        }
        if (exponent->m_requires_grad) {
            std::cerr
                << "Error: gradient cannot be computed for the base in ad::pow"
                << std::endl;
            // NOTE this would just work if we implemented element-wise log for
            // vector/matrix types. But it's too much work for sth that is not
            // going to be used
            // d/dx n^x = n^x log(n) exponent->m_grad =
            // ad::detail::sum(ad::detail::ewise_mult(
            //     v->m_grad, v->m_value * std::log(base->m_value)));
            exponent->backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _Value<T1>* v) -> std::ostream& {
        _Value<T1>* base = static_cast<_Value<T1>*>(v->m_children[0]);
        _Value<T2>* exponent = static_cast<_Value<T2>*>(v->m_children[1]);
        o << *base << "**" << *exponent;
        return o;
    };
    _Value<T1>* result =
        new _Value<T1>(ad::detail::pow(base.m_value, exponent.m_value),
                       backward_f, to_string, "**", {&base, &exponent});
    base.m_parent = result;
    exponent.m_parent = result;
    return *result;
}
AD_POW_TEMPLATE
_Value<T1>& pow(_Value<T1>& lhs, _Value<T2>&& rhs) { return pow(lhs, rhs); }
AD_POW_TEMPLATE
_Value<T1>& pow(_Value<T1>&& lhs, _Value<T2>& rhs) { return pow(lhs, rhs); }
AD_POW_TEMPLATE
_Value<T1>& pow(_Value<T1>&& lhs, _Value<T2>&& rhs) { return pow(lhs, rhs); }
AD_POW_TEMPLATE
_Value<T1>& pow(_Value<T1>& lhs, T2 rhs) {
    return pow(lhs, AD_MAKE_TEMP(rhs, T2));
}
AD_POW_TEMPLATE
_Value<T1>& pow(_Value<T1>&& lhs, T2 rhs) {
    return pow(lhs, AD_MAKE_TEMP(rhs, T2));
}
AD_POW_TEMPLATE
_Value<T1>& pow(T1 lhs, _Value<T2>& rhs) {
    return pow(AD_MAKE_TEMP(lhs, T1), rhs);
}
AD_POW_TEMPLATE
_Value<T1>& pow(T1 lhs, _Value<T2>&& rhs) {
    return pow(AD_MAKE_TEMP(lhs, T1), rhs);
}

/// Element-wise operations ///

template <typename T>
_Value<T>& relu(_Value<T>& obj) {
    static auto backward_f = [](_Value<T>* v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        if (obj->m_requires_grad) {
            obj->m_grad = ad::detail::relu_helper(v->m_value, v->m_grad);
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _Value<T>* v) -> std::ostream& {
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        o << "relu(" << *obj << ")";
        return o;
    };
    _Value<T>* result =
        new _Value<T>(ad::detail::relu_helper(obj.m_value, obj.m_value),
                      backward_f, to_string, "relu", {&obj});
    obj.m_parent = result;
    return *result;
}
template <typename T>
_Value<T>& relu(_Value<T>&& v) {
    return relu(v);
}

template <typename T>
_Value<T>& sigmoid(_Value<T>& obj) {
    static auto backward_f = [](_Value<T>* v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        if (obj->m_requires_grad) {
            obj->m_grad = ad::detail::ewise_mult(
                v->m_grad, ad::detail::ewise_mult(v->m_value, -v->m_value + 1));
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _Value<T>* v) -> std::ostream& {
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        o << "sigmoid(" << *obj << ")";
        return o;
    };
    _Value<T>* result = new _Value<T>(ad::detail::sigmoid(obj.m_value),
                                      backward_f, to_string, "sigmoid", {&obj});
    obj.m_parent = result;
    return *result;
}
template <typename T>
_Value<T>& sigmoid(_Value<T>&& v) {
    return sigmoid(v);
}

template <typename T>
_Value<T>& sin(_Value<T>& obj) {
    static auto backward_f = [](_Value<T>* v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        if (obj->m_requires_grad) {
            obj->m_grad = v->m_grad * ad::detail::cos(obj->m_value);
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _Value<T>* v) -> std::ostream& {
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        o << "sin(" << *obj << ")";
        return o;
    };
    _Value<T>* result = new _Value<T>(ad::detail::sin(obj.m_value), backward_f,
                                      to_string, "sin", {&obj});
    obj.m_parent = result;
    return *result;
}
template <typename T>
_Value<T>& sin(_Value<T>&& v) {
    return sin(v);
}

template <typename T>
_Value<T>& cos(_Value<T>& obj) {
    static auto backward_f = [](_Value<T>* v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        if (obj->m_requires_grad) {
            obj->m_grad = v->m_grad * ad::detail::sin(obj->m_value) * -1.0f;
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _Value<T>* v) -> std::ostream& {
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        o << "cos(" << *obj << ")";
        return o;
    };
    _Value<T>* result = new _Value<T>(ad::detail::cos(obj.m_value), backward_f,
                                      to_string, "cos", {&obj});
    obj.m_parent = result;
    return *result;
}
template <typename T>
_Value<T>& cos(_Value<T>&& v) {
    return cos(v);
}

/// Tensor-like structures ///

template <unsigned int N>
using Vector = _Value<common::Vec<float, N>>;

template <unsigned int N, unsigned int M = N>
using Matrix = _Value<common::Mat<float, N, M>>;

/// Tensor reduce operations ///

template <typename T>
Value& sum(_Value<T>& obj) {
    static auto backward_f = [](Value* v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        if (obj->m_requires_grad) {
            obj->m_grad = v->m_grad;
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const Value* v) -> std::ostream& {
        _Value<T>* obj = static_cast<_Value<T>*>(v->m_children[0]);
        o << "sum(" << *obj << ")";
        return o;
    };
    Value* result = new Value(detail::sum(obj.m_value), backward_f, to_string,
                              "sum", {&obj});
    obj.m_parent = result;
    return *result;
}
template <typename T>
Value& sum(_Value<T>&& v) {
    return sum(v);
}

/// Tensor expand operations ///

template <unsigned int N>
Vector<N>& expand(Value& obj) {
    static auto backward_f = [](Vector<N>* v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        Value* obj = static_cast<Value*>(v->m_children[0]);
        if (obj->m_requires_grad) {
            obj->m_grad = detail::sum(v->m_grad);
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const Vector<N>* v) -> std::ostream& {
        Value* obj = static_cast<Value*>(v->m_children[0]);
        for (unsigned int i = 0; i < N; ++i) o << obj->m_value;
        return o;
    };
    Vector<N>* result =
        new Vector<N>(obj.m_value, backward_f, to_string,
                      "expand(" + std::to_string(N) + ")", {&obj});
    obj.m_parent = result;
    return *result;
}
template <unsigned int N>
Vector<N>& expand(Value&& obj) {
    return expand<N>(obj);
}

template <unsigned int N, unsigned int S>
Vector<S * N>& expand(Vector<S>& obj) {
    static auto backward_f = [](Vector<S * N>* v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        Vector<S>* obj = static_cast<Vector<S>*>(v->m_children[0]);
        if (obj->m_requires_grad) {
            obj->m_grad = 0;
            for (unsigned int i = 0; i < S; ++i)
                for (unsigned int j = i; j < S * N; j += N)
                    obj->m_grad[i] += v->m_grad[j];
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const Vector<S * N>* v) -> std::ostream& {
        Vector<S>* obj = static_cast<Vector<S>*>(v->m_children[0]);
        for (unsigned int i = 0; i < N; ++i) o << obj->m_value;
        return o;
    };
    Vector<S* N>* result =
        new Vector<S * N>(common::Vec<float, S * N>(), backward_f, to_string,
                          "expand(" + std::to_string(N) + ")", {&obj});
    obj.m_parent = result;
    for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < S; ++j)
            result->m_value[i * S + j] = obj.m_value[j];
    return *result;
}
template <unsigned int N, unsigned int S>
Vector<S * N>& expand(Vector<S>&& obj) {
    return expand<N, S>(obj);
}

};  // namespace ad
