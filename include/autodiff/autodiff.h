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
class _Value;
template <typename T, typename = void>
struct is_value : std::false_type {};
template <typename T>
struct is_value<
    T, std::void_t<decltype(static_cast<const _Value<typename T::type>&>(
           std::declval<T>()))>> : std::true_type {};
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

template <typename TR, typename TP, typename TB>
static TR compute_grad_mult(const TP& parent_grad, const TB& brother_value) {
    TR result;
    if constexpr (std::is_scalar_v<TR>) {
        // TB == TP
        result = detail::sum(parent_grad * brother_value);
    } else if constexpr (is_vec_v<TR> && is_mat_v<TB>) {
        result = parent_grad * brother_value.vsum();
    } else if constexpr (is_mat_v<TR> && is_vec_v<TB>) {
        for (unsigned int i = 0; i < TR::rows; ++i)
            for (unsigned int j = 0; j < TR::cols; ++j)
                result(i, j) = parent_grad[i] * brother_value[j];
    } else {
        result = parent_grad * brother_value;
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
        T value, void (*backward_f)(_Value*) = [](_Value<T>*) {},
        std::ostream& (*to_string)(std::ostream&, const _Value<T>*) =
            &_Value<T>::default_to_string,
        std::string op_name = "Value", std::vector<void*> const& children = {})
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

    static _Value* TempValue(T value) {
        _Value* result = new _Value(value);
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

    inline friend std::ostream& operator<<(std::ostream& s, const _Value& v) {
        return v.m_to_string(s, &v);
    }

#define AD_MAKE_TEMP(x, t) *_Value<t>::TempValue(x)
#define AD_TEMPLATE_NON_CLS                                          \
    template <typename T1, typename T2,                              \
              typename = std::enable_if_t<!detail::is_value_v<T1> && \
                                          !detail::is_value_v<T2>>>
// there are many ways to call a simple addition operation
// first, lhs and/or rhs can be lvalue or rvalue (that's why the first 3
// methods are here). second, lhs and/or rhs can be from a class that is not
// _Value. In those cases we make a TempValue as seen above. That's the rest
// of the methods of this list.
// the is_same_template are there to prevent _Value<_Value<T>> recursiveness
#define AD_BINARY_PERFECT_FORWARD(cls, func)            \
    AD_TEMPLATE_NON_CLS                                 \
    friend cls<T>& func(cls<T1>& lhs, cls<T2>&& rhs) {  \
        return func(lhs, rhs);                          \
    }                                                   \
    AD_TEMPLATE_NON_CLS                                 \
    friend cls<T>& func(cls<T1>&& lhs, cls<T2>& rhs) {  \
        return func(lhs, rhs);                          \
    }                                                   \
    AD_TEMPLATE_NON_CLS                                 \
    friend cls<T>& func(cls<T1>&& lhs, cls<T2>&& rhs) { \
        return func(lhs, rhs);                          \
    }                                                   \
    AD_TEMPLATE_NON_CLS                                 \
    friend cls<T>& func(cls<T1>& lhs, T2 rhs) {         \
        return func(lhs, AD_MAKE_TEMP(rhs, T2));        \
    }                                                   \
    AD_TEMPLATE_NON_CLS                                 \
    friend cls<T>& func(cls<T1>&& lhs, T2 rhs) {        \
        return func(lhs, AD_MAKE_TEMP(rhs, T2));        \
    }                                                   \
    AD_TEMPLATE_NON_CLS                                 \
    friend cls<T>& func(T1 lhs, cls<T2>& rhs) {         \
        return func(AD_MAKE_TEMP(lhs, T1), rhs);        \
    }                                                   \
    AD_TEMPLATE_NON_CLS                                 \
    friend cls<T>& func(T1 lhs, cls<T2>&& rhs) {        \
        return func(AD_MAKE_TEMP(lhs, T1), rhs);        \
    }

#define AD_BINARY_OP(op, lhs_grad, rhs_grad)                                 \
    template <typename T1, typename T2,                                      \
              typename = std::enable_if_t<std::is_same_v<                    \
                  T, decltype(std::declval<T1>() op std::declval<T2>())>>>   \
    friend _Value<T>& operator op(_Value<T1>& lhs, _Value<T2>& rhs) {        \
        static auto backward_f = [](_Value<T>* v) {                          \
            AD_ENSURE_REQUIRES_GRAD(v);                                      \
            _Value<T1>* lhs = static_cast<_Value<T1>*>(v->m_children[0]);    \
            _Value<T2>* rhs = static_cast<_Value<T2>*>(v->m_children[1]);    \
            if (lhs->m_requires_grad) {                                      \
                lhs->m_grad = lhs_grad;                                      \
                lhs->backward();                                             \
            }                                                                \
            if (rhs->m_requires_grad) {                                      \
                rhs->m_grad = rhs_grad;                                      \
                rhs->backward();                                             \
            }                                                                \
        };                                                                   \
        static auto to_string = [](std::ostream& o,                          \
                                   const _Value<T>* v) -> std::ostream& {    \
            _Value<T1>* lhs = static_cast<_Value<T1>*>(v->m_children[0]);    \
            _Value<T2>* rhs = static_cast<_Value<T2>*>(v->m_children[1]);    \
            o << *lhs << #op << *rhs;                                        \
            return o;                                                        \
        };                                                                   \
        _Value<T>* result =                                                  \
            new _Value<T>(lhs.m_value op rhs.m_value, backward_f, to_string, \
                          #op, {&lhs, &rhs});                                \
        return *result;                                                      \
    }                                                                        \
    AD_BINARY_PERFECT_FORWARD(_Value, operator op)

    AD_BINARY_OP(+, detail::sum_if_scalar<decltype(lhs->m_value)>(v->m_grad),
                 detail::sum_if_scalar<decltype(rhs->m_value)>(v->m_grad));
    AD_BINARY_OP(-, detail::sum_if_scalar<decltype(lhs->m_value)>(v->m_grad),
                 -1.0f *
                     detail::sum_if_scalar<decltype(rhs->m_value)>(v->m_grad));
    // NOTE this is the old pre-tensor version
    // AD_BINARY_OP(*, lhs.m_value *rhs.m_value, v->m_grad * rhs->m_value,
    //              v->m_grad * lhs->m_value);
    AD_BINARY_OP(*, detail::compute_grad_mult<T1>(v->m_grad, rhs->m_value),
                 detail::compute_grad_mult<T2>(v->m_grad, lhs->m_value));
    // TODO create compute_grad_div or sth and enable division
    // AD_BINARY_OP(/, lhs.m_value / rhs.m_value, v->m_grad / rhs->m_value,
    //              v->m_grad * lhs->m_value / (rhs->m_value * rhs->m_value));

#define AD_UNARY_OP(op, value, grad)                                      \
    template <typename T1>                                                \
    friend _Value<T>& operator op(_Value<T1>& obj) {                      \
        static auto backward_f = [](_Value<T>* v) {                       \
            AD_ENSURE_REQUIRES_GRAD(v);                                   \
            _Value<T1>* obj = static_cast<_Value<T1>*>(v->m_children[0]); \
            if (obj->m_requires_grad) {                                   \
                obj->m_grad = grad;                                       \
                obj->backward();                                          \
            }                                                             \
        };                                                                \
        static auto to_string = [](std::ostream& o,                       \
                                   const _Value<T>* v) -> std::ostream& { \
            _Value<T1>* obj = static_cast<_Value<T1>*>(v->m_children[0]); \
            o << #op << *obj;                                             \
            return o;                                                     \
        };                                                                \
        _Value<T>* result =                                               \
            new _Value<T>(value, backward_f, to_string, #op, {&obj});     \
        return *result;                                                   \
    }                                                                     \
    friend _Value<T>& operator op(_Value<T>&& obj) { return operator op(obj); }

    AD_UNARY_OP(-, -obj.m_value, v->m_grad * -1);

    template <typename T2>
    friend _Value<T2>& pow(_Value<T2>& base, _Value<T2>& exponent);
    template <typename T2>
    friend _Value<T2>& relu(_Value<T2>& obj);
    template <typename T2>
    friend _Value<float>& sum(_Value<T2>& obj);
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
                              "relu", {&obj});
    return *result;
}
template <typename T>
Value& sum(_Value<T>&& v) {
    return sum(v);
}

};  // namespace ad
