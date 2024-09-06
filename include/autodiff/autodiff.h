#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "libcpp-common/geometry.h"

namespace ad {

// check if two types come from the same template
// e.g. Value<float> and Value<int>
template <typename, typename>
struct is_same_template : std::false_type {};
template <template <typename...> class C, typename... Args1, typename... Args2>
struct is_same_template<C<Args1...>, C<Args2...>> : std::true_type {};
template <typename T1, typename T2>
constexpr bool is_same_template_v = is_same_template<T1, T2>::value;

template <typename T>
struct is_vec : std::false_type {};

template <typename U, unsigned int N>
struct is_vec<common::Vec<U, N>> : std::true_type {};

template <typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

template <typename TR, typename TP, typename TB>
static TR compute_grad_mult(const TP &parent_grad, const TB &brother_grad) {
    TR result;
    if constexpr (std::is_scalar_v<TB>) {
        result = parent_grad * brother_grad;
    } else if constexpr (is_vec_v<TB>) {
        // TODO if TR is a scalar, then result = sum(parent * brother)
        // if TR is another vector, result = parent * brother
        // if TR is a matrix, i don't know yet
    }  // TODO check is_mat_v and if it's not the case raise assertion error
    return result;
}

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

    template <typename U>
    friend class _Value;

   private:
    T m_value;
    T m_grad;
    bool m_has_grad;
    bool m_requires_grad;
    void (*m_backward_f)(_Value<T> *);
    std::ostream &(*m_to_string)(std::ostream &, const _Value<T> *);
    std::string m_op_name;
    // children/parent can be of different T
    void *m_parent;
    std::vector<void *> m_children;

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
        std::string op_name = "Value", std::vector<void *> const &children = {})
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

    inline friend std::ostream &operator<<(std::ostream &s, const _Value &v) {
        return v.m_to_string(s, &v);
    }

#define AD_MAKE_TEMP(x, t) *_Value<t>::TempValue(x)
// there are many ways to call a simple addition operation
// first, lhs and/or rhs can be lvalue or rvalue (that's why the first 3
// methods are here). second, lhs and/or rhs can be from a class that is not
// _Value. In those cases we make a TempValue as seen above. That's the rest
// of the methods of this list.
// the is_same_template are there to prevent _Value<_Value<T>> recursiveness
#define AD_BINARY_PERFECT_FORWARD(cls, func)                              \
    template <typename T1, typename T2,                                   \
              typename = std::enable_if_t<!is_same_template_v<T1, cls> && \
                                          !is_same_template_v<T2, cls>>>  \
    friend cls<T> &func(cls<T1> &lhs, cls<T2> &&rhs) {                    \
        return func(lhs, rhs);                                            \
    }                                                                     \
    template <typename T1, typename T2,                                   \
              typename = std::enable_if_t<!is_same_template_v<T1, cls> && \
                                          !is_same_template_v<T2, cls>>>  \
    friend cls<T> &func(cls<T1> &&lhs, cls<T2> &rhs) {                    \
        return func(lhs, rhs);                                            \
    }                                                                     \
    template <typename T1, typename T2,                                   \
              typename = std::enable_if_t<!is_same_template_v<T1, cls> && \
                                          !is_same_template_v<T2, cls>>>  \
    friend cls<T> &func(cls<T1> &&lhs, cls<T2> &&rhs) {                   \
        return func(lhs, rhs);                                            \
    }                                                                     \
    template <typename T1, typename T2,                                   \
              typename = std::enable_if_t<!is_same_template_v<T1, cls> && \
                                          !is_same_template_v<T2, cls>>>  \
    friend cls<T> &func(cls<T1> &lhs, T2 rhs) {                           \
        return func(lhs, AD_MAKE_TEMP(rhs, T2));                          \
    }                                                                     \
    template <typename T1, typename T2,                                   \
              typename = std::enable_if_t<!is_same_template_v<T1, cls> && \
                                          !is_same_template_v<T2, cls>>>  \
    friend cls<T> &func(cls<T1> &&lhs, T2 rhs) {                          \
        return func(lhs, AD_MAKE_TEMP(rhs, T2));                          \
    }                                                                     \
    template <typename T1, typename T2,                                   \
              typename = std::enable_if_t<!is_same_template_v<T1, cls> && \
                                          !is_same_template_v<T2, cls>>>  \
    friend cls<T> &func(T1 lhs, cls<T2> &rhs) {                           \
        return func(AD_MAKE_TEMP(lhs, T1), rhs);                          \
    }                                                                     \
    template <typename T1, typename T2,                                   \
              typename = std::enable_if_t<!is_same_template_v<T1, cls> && \
                                          !is_same_template_v<T2, cls>>>  \
    friend cls<T> &func(T1 lhs, cls<T2> &&rhs) {                          \
        return func(AD_MAKE_TEMP(lhs, T1), rhs);                          \
    }

#define AD_BINARY_OP(op, lhs_grad, rhs_grad)                                 \
    template <typename T1, typename T2,                                      \
              typename = std::enable_if_t<std::is_same_v<                    \
                  T, decltype(std::declval<T1>() op std::declval<T2>())>>>   \
    friend _Value<T> &operator op(_Value<T1> &lhs, _Value<T2> &rhs) {        \
        static auto backward_f = [](_Value<T> *v) {                          \
            AD_ENSURE_REQUIRES_GRAD(v);                                      \
            _Value<T1> *lhs = static_cast<_Value<T1> *>(v->m_children[0]);   \
            _Value<T2> *rhs = static_cast<_Value<T2> *>(v->m_children[1]);   \
            if (lhs->m_requires_grad) {                                      \
                lhs->m_grad = lhs_grad;                                      \
                lhs->backward();                                             \
            }                                                                \
            if (rhs->m_requires_grad) {                                      \
                rhs->m_grad = rhs_grad;                                      \
                rhs->backward();                                             \
            }                                                                \
        };                                                                   \
        static auto to_string = [](std::ostream &o,                          \
                                   const _Value<T> *v) -> std::ostream & {   \
            _Value<T1> *lhs = static_cast<_Value<T1> *>(v->m_children[0]);   \
            _Value<T2> *rhs = static_cast<_Value<T2> *>(v->m_children[1]);   \
            o << *lhs << #op << *rhs;                                        \
            return o;                                                        \
        };                                                                   \
        _Value<T> *result =                                                  \
            new _Value<T>(lhs.m_value op rhs.m_value, backward_f, to_string, \
                          #op, {&lhs, &rhs});                                \
        return *result;                                                      \
    }                                                                        \
    AD_BINARY_PERFECT_FORWARD(_Value, operator op)

    // TODO these probably also need custom grad calculation (sum gradient of
    // parent if it's a vector?????)
    AD_BINARY_OP(+, v->m_grad, v->m_grad);
    AD_BINARY_OP(-, v->m_grad, v->m_grad * -1.0f);
    // NOTE this is the old pre-tensor version
    // AD_BINARY_OP(*, lhs.m_value *rhs.m_value, v->m_grad * rhs->m_value,
    //              v->m_grad * lhs->m_value);
    AD_BINARY_OP(*, compute_grad_mult<T1>(v->m_grad, rhs->m_value),
                 compute_grad_mult<T2>(v->m_grad, lhs->m_value));
    // TODO create compute_grad_div or sth and enable division
    // AD_BINARY_OP(/, lhs.m_value / rhs.m_value, v->m_grad / rhs->m_value,
    //              v->m_grad * lhs->m_value / (rhs->m_value * rhs->m_value));

#define AD_UNARY_OP(op, value, grad)                                       \
    template <typename T1>                                                 \
    friend _Value<T> &operator op(_Value<T1> &obj) {                       \
        static auto backward_f = [](_Value<T> *v) {                        \
            AD_ENSURE_REQUIRES_GRAD(v);                                    \
            _Value<T1> *obj = static_cast<_Value<T1> *>(v->m_children[0]); \
            if (obj->m_requires_grad) {                                    \
                obj->m_grad = grad;                                        \
                obj->backward();                                           \
            }                                                              \
        };                                                                 \
        static auto to_string = [](std::ostream &o,                        \
                                   const _Value<T> *v) -> std::ostream & { \
            _Value<T1> *obj = static_cast<_Value<T1> *>(v->m_children[0]); \
            o << #op << *obj;                                              \
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
    template <typename T2>
    friend _Value<float> &sum(_Value<T2> &obj);
};

using Value = _Value<float>;

/// Math functions ///

// template <typename T, typename T1, typename T2>
// _Value<T> &pow(_Value<T1> &base, _Value<T2> &exponent) {
//     static auto backward_f = [](_Value<T> *v) {
//         AD_ENSURE_REQUIRES_GRAD(v);
//         auto base = v->m_children[0], exponent = v->m_children[1];
//         if (base->m_requires_grad) {
//             // d/dx x^n = n*x^(n-1)
//             base->m_grad = v->m_grad * exponent->m_value *
//                            std::pow(base->m_value, exponent->m_value - 1.0f);
//             base->backward();
//         }
//         if (exponent->m_requires_grad) {
//             // d/dx n^x = n^x log(n)
//             exponent->m_grad = v->m_grad * v->m_value *
//             std::log(base->m_value); exponent->backward();
//         }
//     };
//     static auto to_string = [](std::ostream &o,
//                                const _Value<T> *v) -> std::ostream & {
//         o << *v->m_children[0] << "**" << *v->m_children[1];
//         return o;
//     };
//     _Value<T> *result =
//         new _Value<T>(std::pow(base.m_value, exponent.m_value), backward_f,
//                       to_string, "**", {&base, &exponent});
//     return *result;
// }
// // TODO this will need another perfect forward macro probably
// // separate from binary operations in the class
// AD_BINARY_PERFECT_FORWARD(_Value, pow);

/// Element-wise operations ///

// template <typename T>
// _Value<T> &relu(_Value<T> &obj) {
//     static auto backward_f = [](_Value<T> *v) {
//         AD_ENSURE_REQUIRES_GRAD(v);
//         auto obj = v->m_children[0];
//         if (obj->m_requires_grad) {
//             obj->m_grad = v->m_value > 0 ? v->m_grad : 0;
//             obj->backward();
//         }
//     };
//     static auto to_string = [](std::ostream &o,
//                                const _Value<T> *v) -> std::ostream & {
//         o << "relu(" << *v->m_children[0] << ")";
//         return o;
//     };
//     _Value<T> *result = new _Value<T>(obj.m_value > 0 ? obj.m_value : 0,
//                                       backward_f, to_string, "relu", {&obj});
//     return *result;
// }
// template <typename T>
// _Value<T> &relu(_Value<T> &&v) {
//     return relu(v);
// }

/// Tensor-like structures ///

template <unsigned int N>
class Vector : public _Value<common::Vec<float, N>> {
   private:
    using Base = _Value<common::Vec<float, N>>;

    // helper to unpack each element in array
    template <std::size_t... I>
    Vector(const std::array<float, N> &values, std::index_sequence<I...>)
        : Base(common::Vec<float, N>(values[I]...)) {}

   public:
    COMMON_VEC_IMPORT(Vector, Base);

    // constructor with std::array Vector(std::array{1,2,3})
    explicit Vector(const std::array<float, N> &values)
        : Vector(values, std::make_index_sequence<N>{}) {}
};

/// Tensor reduce operations ///

template <typename T>
Value &sum(_Value<T> &obj) {
    static auto backward_f = [](Value *v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _Value<T> *obj = static_cast<_Value<T> *>(v->m_children[0]);
        if (obj->m_requires_grad) {
            obj->m_grad = v->m_grad;
            obj->backward();
        }
    };
    static auto to_string = [](std::ostream &o,
                               const Value *v) -> std::ostream & {
        _Value<T> *obj = static_cast<_Value<T> *>(v->m_children[0]);
        o << "sum(" << *obj << ")";
        return o;
    };
    float sum_value = 0;
    if constexpr (std::is_scalar_v<T>) {
        sum_value = obj.m_value;
    } else if constexpr (is_vec_v<T>) {
        for (size_t i = 0; i < T::size; ++i) sum_value += obj.m_value[i];
    }
    Value *result = new Value(sum_value, backward_f, to_string, "relu", {&obj});
    return *result;
}
template <typename T>
Value &sum(_Value<T> &&v) {
    return sum(v);
}

};  // namespace ad
