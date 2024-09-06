#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "libcpp-common/geometry.h"

namespace ad {

#define AD_ENSURE_REQUIRES_GRAD(x)                                       \
    if (!x->m_requires_grad) {                                           \
        std::cerr << "Tried calling backward on a node without gradient" \
                  << std::endl;                                          \
        return;                                                          \
    }

template <typename T>
class _Value {
   public:
    using type = T;

   private:
    T m_value;
    T m_grad;
    bool m_has_grad;
    bool m_requires_grad;
    void (*m_backward_f)(_Value<T> *);
    std::ostream &(*m_to_string)(std::ostream &, const _Value<T> *);
    std::string m_op_name;
    _Value<T> *m_parent;
    std::vector<_Value<T> *> m_children;

    static std::ostream &default_to_string(std::ostream &o,
                                           const _Value<T> *v) {
        o << v->m_value;
        return o;
    }

   public:
    explicit _Value(
        T value, void (*backward_f)(_Value *) = [](_Value<T> *) {},
        std::ostream &(*to_string)(std::ostream &, const _Value<T> *) =
            &_Value<T>::default_to_string,
        std::string op_name = "Value",
        std::vector<_Value *> const &children = {})
        : m_value(value),
          m_grad(1),
          m_has_grad(false),
          m_requires_grad(true),
          m_backward_f(backward_f),
          m_to_string(to_string),
          m_op_name(op_name),
          m_parent(nullptr),
          m_children(children) {}
    // TODO handle parent node in destructor (destroy it)
    ~_Value() { std::cerr << "Destructor called" << std::endl; }

    static _Value *TempValue(T value) {
        _Value *result = new _Value(value);
        result->m_requires_grad = false;
        return result;
    }

    void backward() {
        m_has_grad = true;
        m_backward_f(this);
    }
    T value() const { return m_value; }
    T grad() const {
        if (!m_has_grad) {
            std::cerr << "grad() called on a node without computed gradient"
                      << std::endl;
        }
        return m_grad;
    }

    friend std::ostream &operator<<(std::ostream &s, const _Value &v) {
        return v.m_to_string(s, &v);
    }

#define AD_MAKE_TEMP(x, t) *_Value<t>::TempValue(x)
#define AD_BINARY_PERFECT_FORWARD(cls, before, func)                  \
    before cls &func(cls &lhs, cls &&rhs) { return func(lhs, rhs); }  \
    before cls &func(cls &&lhs, cls &rhs) { return func(lhs, rhs); }  \
    before cls &func(cls &&lhs, cls &&rhs) { return func(lhs, rhs); } \
    before cls &func(cls &lhs, T rhs) {                               \
        return func(lhs, AD_MAKE_TEMP(rhs, T));                       \
    }                                                                 \
    before cls &func(cls &&lhs, T rhs) {                              \
        return func(lhs, AD_MAKE_TEMP(rhs, T));                       \
    }                                                                 \
    before cls &func(T lhs, cls &rhs) {                               \
        return func(AD_MAKE_TEMP(lhs, T), rhs);                       \
    }                                                                 \
    before cls &func(T lhs, cls &&rhs) {                              \
        return func(AD_MAKE_TEMP(lhs, T), rhs);                       \
    }

#define AD_BINARY_OP(op, value, lhs_grad, rhs_grad)                         \
    friend _Value<T> &operator op(_Value<T> &lhs, _Value<T> &rhs) {         \
        static auto backward_f = [](_Value<T> *v) {                         \
            AD_ENSURE_REQUIRES_GRAD(v);                                     \
            auto lhs = v->m_children[0], rhs = v->m_children[1];            \
            if (lhs->m_requires_grad) {                                     \
                lhs->m_grad = lhs_grad;                                     \
                lhs->backward();                                            \
            }                                                               \
            if (rhs->m_requires_grad) {                                     \
                rhs->m_grad = rhs_grad;                                     \
                rhs->backward();                                            \
            }                                                               \
        };                                                                  \
        static auto to_string = [](std::ostream &o,                         \
                                   const _Value<T> *v) -> std::ostream & {  \
            o << *v->m_children[0] << #op << *v->m_children[1];             \
            return o;                                                       \
        };                                                                  \
        _Value<T> *result =                                                 \
            new _Value<T>(value, backward_f, to_string, #op, {&lhs, &rhs}); \
        return *result;                                                     \
    }                                                                       \
    AD_BINARY_PERFECT_FORWARD(_Value, friend, operator op)

    AD_BINARY_OP(+, lhs.m_value + rhs.m_value, v->m_grad, v->m_grad);
    AD_BINARY_OP(-, lhs.m_value - rhs.m_value, v->m_grad, v->m_grad * -1.0f);
    AD_BINARY_OP(*, lhs.m_value *rhs.m_value, v->m_grad * rhs->m_value,
                 v->m_grad * lhs->m_value);
    AD_BINARY_OP(/, lhs.m_value / rhs.m_value, v->m_grad / rhs->m_value,
                 v->m_grad * lhs->m_value / (rhs->m_value * rhs->m_value));

#define AD_UNARY_OP(op, value, grad)                                       \
    friend _Value<T> &operator op(_Value<T> &obj) {                        \
        static auto backward_f = [](_Value<T> *v) {                        \
            AD_ENSURE_REQUIRES_GRAD(v);                                    \
            auto obj = v->m_children[0];                                   \
            if (obj->m_requires_grad) {                                    \
                obj->m_grad = grad;                                        \
                obj->backward();                                           \
            }                                                              \
        };                                                                 \
        static auto to_string = [](std::ostream &o,                        \
                                   const _Value<T> *v) -> std::ostream & { \
            o << #op << *v->m_children[0];                                 \
            return o;                                                      \
        };                                                                 \
        _Value<T> *result =                                                \
            new _Value<T>(value, backward_f, to_string, #op, {&obj});      \
        return *result;                                                    \
    }                                                                      \
    friend _Value<T> &operator op(_Value<T> &&obj) { return operator op(obj); }

    AD_UNARY_OP(-, -obj.m_value, v->m_grad * -1);

    template <typename T2>
    friend _Value<T2> &pow(_Value<T2> &base, _Value<T2> &exponent);
    template <typename T2>
    friend _Value<T2> &relu(_Value<T2> &obj);
};

using Value = _Value<float>;

/// Math functions ///

template <typename T>
_Value<T> &pow(_Value<T> &base, _Value<T> &exponent) {
    static auto backward_f = [](_Value<T> *v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        auto base = v->m_children[0], exponent = v->m_children[1];
        if (base->m_requires_grad) {
            // d/dx x^n = n*x^(n-1)
            base->m_grad = v->m_grad * exponent->m_value *
                           std::pow(base->m_value, exponent->m_value - 1.0f);
            base->backward();
        }
        if (exponent->m_requires_grad) {
            // d/dx n^x = n^x log(n)
            exponent->m_grad = v->m_grad * v->m_value * std::log(base->m_value);
            exponent->backward();
        }
    };
    static auto to_string = [](std::ostream &o,
                               const _Value<T> *v) -> std::ostream & {
        o << *v->m_children[0] << "**" << *v->m_children[1];
        return o;
    };
    _Value<T> *result =
        new _Value<T>(std::pow(base.m_value, exponent.m_value), backward_f,
                      to_string, "**", {&base, &exponent});
    return *result;
}
AD_BINARY_PERFECT_FORWARD(_Value<T>, template <typename T>, pow);

/// NN related functions ///

template <typename T>
_Value<T> &relu(_Value<T> &obj) {
    static auto backward_f = [](_Value<T> *v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        auto obj = v->m_children[0];
        if (obj->m_requires_grad) {
            obj->m_grad = v->m_value > 0 ? v->m_grad : 0;
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream &o,
                               const _Value<T> *v) -> std::ostream & {
        o << "relu(" << *v->m_children[0] << ")";
        return o;
    };
    _Value<T> *result = new _Value<T>(obj.m_value > 0 ? obj.m_value : 0,
                                      backward_f, to_string, "relu", {&obj});
    return *result;
}
template <typename T>
_Value<T> &relu(_Value<T> &&v) {
    return relu(v);
}

/// Tensor-like structures ///

template <unsigned int N>
class Vector : public _Value<common::Vec<float, N>> {
   private:
    using Base = _Value<common::Vec<float, N>>;

    template <std::size_t... I>
    Vector(const std::array<float, N> &values, std::index_sequence<I...>)
        : Base(common::Vec<float, N>(values[I]...)) {}

   public:
    COMMON_VEC_IMPORT(Vector, Base);

    explicit Vector(const std::array<float, N> &values)
        : Vector(values, std::make_index_sequence<N>{}) {}
};

};  // namespace ad
