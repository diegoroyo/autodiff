#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "autodiff/types.h"
#include "autodiff/util.h"

namespace ad {

template <typename T>
class _ValueWrapper;

using Value = _ValueWrapper<float>;

template <unsigned int N>
using Vector = _ValueWrapper<common::Vec<float, N>>;

template <unsigned int N, unsigned int M = N>
using Matrix = _ValueWrapper<common::Mat<float, N, M>>;

#define AD_CLASS_FUNCTIONS                                   \
    template <typename B, typename E, typename>              \
    friend _ValueWrapper<B> pow(_ValueWrapper<B>& base,      \
                                _ValueWrapper<E>& exponent); \
    template <typename A>                                    \
    friend _ValueWrapper<A> relu(_ValueWrapper<A>& obj);     \
    template <typename A>                                    \
    friend _ValueWrapper<A> sigmoid(_ValueWrapper<A>& obj);  \
    template <typename A>                                    \
    friend Value sum(_ValueWrapper<A>& obj);                 \
    template <typename A>                                    \
    friend _ValueWrapper<A> sin(_ValueWrapper<A>& obj);      \
    template <typename A>                                    \
    friend _ValueWrapper<A> cos(_ValueWrapper<A>& obj);      \
    template <unsigned int N>                                \
    friend Vector<N> expand(Value& obj);                     \
    template <unsigned int N, unsigned int S>                \
    friend Vector<S * N> expand(Vector<S>& obj);

class _AbstractValue {};

template <typename T, typename = std::enable_if_t<std::is_scalar_v<T> ||
                                                  detail::is_vec_v<T> ||
                                                  detail::is_mat_v<T>>>
class _ValueData : public _AbstractValue {
   public:
    template <typename A, typename>
    friend class _ValueData;
    template <typename A>
    friend class _ValueWrapper;

    template <typename A>
    _ValueData<A>& get_child(size_t i) const {
        return *std::static_pointer_cast<_ValueData<A>>(m_children[i]);
    }

   private:
    T m_value;
    T m_grad;
    bool m_has_grad;
    bool m_requires_grad;
    void (*m_backward_f)(_ValueData<T>&);
    std::ostream& (*m_to_string)(std::ostream&, const _ValueData<T>&);
    std::string m_op_name;
    _AbstractValue* m_parent;
    std::vector<std::shared_ptr<_AbstractValue>> m_children;

    _ValueData(T value, void (*backward_f)(_ValueData<T>&),
               std::ostream& (*to_string)(std::ostream&, const _ValueData<T>&),
               std::string op_name,
               std::vector<std::shared_ptr<_AbstractValue>> const& children)
        : m_value(value),
          m_grad(1.0f),
          m_has_grad(false),
          m_requires_grad(true),
          m_backward_f(backward_f),
          m_to_string(to_string),
          m_op_name(op_name),
          m_parent(nullptr),
          m_children(children) {}

    void backward() {
        m_has_grad = true;
        m_backward_f(*this);
    }
    T& value() { return m_value; }
    T& grad() {
        if (!m_has_grad) {
            throw ad::ADException(
                "grad() called on a node without computed gradient");
        }
        return m_grad;
    }
    void update(float lr) { m_value -= grad() * lr; }

    friend std::ostream& operator<<(std::ostream& o, const _ValueData<T>& v) {
        return v.m_to_string(o, v);
    }

    static std::ostream& default_to_string(std::ostream& o,
                                           const _ValueData<T>& v) {
        o << v.m_value;
        return o;
    }

    AD_CLASS_FUNCTIONS
};

template <typename T>
class _ValueWrapper {
   public:
    using _ad_value_type = T;
    using type = T;

   private:
    std::shared_ptr<_ValueData<T>> m_ptr;

    template <typename A, typename>
    friend class _ValueData;
    template <typename A>
    friend class _ValueWrapper;

    template <typename A>
    void set_parent(const _ValueWrapper<A>& v) {
        m_ptr->m_parent =
            std::static_pointer_cast<_AbstractValue>(v.m_ptr).get();
    }

   public:
    explicit _ValueWrapper(
        T value, void (*backward_f)(_ValueData<T>&) = [](_ValueData<T>&) {},
        std::ostream& (*to_string)(std::ostream&, const _ValueData<T>&) =
            &_ValueData<T>::default_to_string,
        std::string op_name = "Value",
        std::vector<std::shared_ptr<_AbstractValue>> const& children = {})
        : m_ptr(new _ValueData<T>(value, backward_f, to_string, op_name,
                                  children)) {}

    static _ValueWrapper<T> TempValue(T x) {
        _ValueWrapper<T> result(x);
        result.m_ptr->m_requires_grad = false;
        return result;
    }

    void backward() { m_ptr->backward(); }
    T& value() { return m_ptr->value(); }
    T& grad() { return m_ptr->grad(); }
    void update(float lr) { m_ptr->update(lr); }
    bool requires_grad() const { return m_ptr->m_requires_grad; }

    inline friend std::ostream& operator<<(std::ostream& o,
                                           const _ValueWrapper& v) {
        o << *v.m_ptr;
        return o;
    }

    /// Binary operators - this requires some macros... ///

#define AD_CHILD(x) x.m_ptr
#define AD_MAKE_TEMP(x, A) _ValueWrapper<A>::TempValue(x)
#define AD_TEMPLATE_NON_CLS(op)                                     \
    template <typename B,                                           \
              typename = std::enable_if_t<!detail::is_value_v<T> && \
                                          !detail::is_value_v<B>>,  \
              typename R = decltype(std::declval<T>() op std::declval<B>())>
    // there are many ways to call a simple addition operation
    // first, lhs and/or rhs can be lvalue or rvalue (that's why the first 3
    // methods are here). second, lhs and/or rhs can be from a class that is not
    // _Value. In those cases we make a TempValue as seen above. That's the rest
    // of the methods of this list.
    // the is_same_template are there to prevent _Value<_Value<T>> recursiveness
#define AD_BINARY_OP(op, lhs_grad, rhs_grad)                                  \
    AD_TEMPLATE_NON_CLS(op)                                                   \
    friend _ValueWrapper<R> operator op(_ValueWrapper<T>& lhs,                \
                                        _ValueWrapper<B>& rhs) {              \
        static auto backward_f = [](_ValueData<R>& v) {                       \
            AD_ENSURE_REQUIRES_GRAD(v);                                       \
            _ValueData<T>& lhs = v.template get_child<T>(0);                  \
            _ValueData<B>& rhs = v.template get_child<B>(1);                  \
            if (lhs.m_requires_grad) {                                        \
                lhs.m_grad = lhs_grad;                                        \
                lhs.backward();                                               \
            }                                                                 \
            if (rhs.m_requires_grad) {                                        \
                rhs.m_grad = rhs_grad;                                        \
                rhs.backward();                                               \
            }                                                                 \
        };                                                                    \
        static auto to_string = [](std::ostream& o,                           \
                                   const _ValueData<R>& v) -> std::ostream& { \
            _ValueData<T>& lhs = v.template get_child<T>(0);                  \
            _ValueData<B>& rhs = v.template get_child<B>(1);                  \
            o << lhs << #op << rhs;                                           \
            return o;                                                         \
        };                                                                    \
        _ValueWrapper<R> result(lhs.value() op rhs.value(), backward_f,       \
                                to_string, #op,                               \
                                {AD_CHILD(lhs), AD_CHILD(rhs)});              \
        lhs.set_parent(result);                                               \
        rhs.set_parent(result);                                               \
        return result;                                                        \
    }                                                                         \
    AD_TEMPLATE_NON_CLS(op)                                                   \
    friend _ValueWrapper<R> operator op(_ValueWrapper<T>& lhs,                \
                                        _ValueWrapper<B>&& rhs) {             \
        return operator op(lhs, rhs);                                         \
    }                                                                         \
    AD_TEMPLATE_NON_CLS(op)                                                   \
    friend _ValueWrapper<R> operator op(_ValueWrapper<T>&& lhs,               \
                                        _ValueWrapper<B>& rhs) {              \
        return operator op(lhs, rhs);                                         \
    }                                                                         \
    AD_TEMPLATE_NON_CLS(op)                                                   \
    friend _ValueWrapper<R> operator op(_ValueWrapper<T>&& lhs,               \
                                        _ValueWrapper<B>&& rhs) {             \
        return operator op(lhs, rhs);                                         \
    }                                                                         \
    AD_TEMPLATE_NON_CLS(op)                                                   \
    friend _ValueWrapper<R> operator op(_ValueWrapper<T>& lhs, B rhs) {       \
        return operator op(lhs, AD_MAKE_TEMP(rhs, B));                        \
    }                                                                         \
    AD_TEMPLATE_NON_CLS(op)                                                   \
    friend _ValueWrapper<R> operator op(_ValueWrapper<T>&& lhs, B rhs) {      \
        return operator op(lhs, AD_MAKE_TEMP(rhs, B));                        \
    }                                                                         \
    AD_TEMPLATE_NON_CLS(op)                                                   \
    friend _ValueWrapper<R> operator op(T lhs, _ValueWrapper<B>& rhs) {       \
        return operator op(AD_MAKE_TEMP(lhs, T), rhs);                        \
    }                                                                         \
    AD_TEMPLATE_NON_CLS(op)                                                   \
    friend _ValueWrapper<R> operator op(T lhs, _ValueWrapper<B>&& rhs) {      \
        return operator op(AD_MAKE_TEMP(lhs, T), rhs);                        \
    }

    // sorry for so many macros. finally here are the damn operators
    AD_BINARY_OP(+, detail::sum_if_scalar<decltype(lhs.m_value)>(v.m_grad),
                 detail::sum_if_scalar<decltype(rhs.m_value)>(v.m_grad));
    AD_BINARY_OP(-, detail::sum_if_scalar<decltype(lhs.m_value)>(v.m_grad),

                 detail::sum_if_scalar<decltype(rhs.m_value)>(v.m_grad) *
                     -1.0f);
    AD_BINARY_OP(*, detail::compute_grad_mult<T>(v.m_grad, rhs.m_value),
                 detail::compute_grad_mult<B>(v.m_grad, lhs.m_value));
    AD_BINARY_OP(/,
                 detail::sum_if_scalar<decltype(lhs.m_value)>(v.m_grad /
                                                              rhs.m_value),
                 detail::sum_if_scalar<decltype(rhs.m_value)>(
                     v.m_grad* lhs.m_value / (rhs.m_value * rhs.m_value)));

    /// Unary operators ///

#define AD_UNARY_OP(op, value, grad)                                          \
    template <typename R = decltype(op std::declval<T>())>                    \
    friend _ValueWrapper<R> operator op(_ValueWrapper<T>& obj) {              \
        static auto backward_f = [](_ValueData<R>& v) {                       \
            AD_ENSURE_REQUIRES_GRAD(v);                                       \
            _ValueData<T>& child = v.template get_child<T>(0);                \
            if (child.m_requires_grad) {                                      \
                child.m_grad = grad;                                          \
                child.backward();                                             \
            }                                                                 \
        };                                                                    \
        static auto to_string = [](std::ostream& o,                           \
                                   const _ValueData<R>& v) -> std::ostream& { \
            o << #op << v.template get_child<T>(0);                           \
            return o;                                                         \
        };                                                                    \
        _ValueWrapper<R> result(value, backward_f, to_string, #op,            \
                                {AD_CHILD(obj)});                             \
        obj.set_parent(result);                                               \
        return result;                                                        \
    }                                                                         \
    template <typename R = decltype(op std::declval<T>())>                    \
    friend _ValueWrapper<R> operator op(_ValueWrapper<T>&& obj) {             \
        return operator op(obj);                                              \
    }

    AD_UNARY_OP(-, -obj.value(), v.m_grad * -1);

    /// Other functions ///

    AD_CLASS_FUNCTIONS
};

};  // namespace ad

#include "value.tpp"