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

class Value {
   private:
    float m_value;
    float m_grad;
    bool m_has_grad;
    bool m_requires_grad;
    void (*m_backward_f)(Value *);
    std::string (*m_to_string)(Value *);
    std::string m_op_name;
    Value *m_parent;
    std::vector<Value *> m_children;

   public:
    explicit Value(
        float value, void (*backward_f)(Value *) = [](Value *) {},
        std::string (*to_string)(Value *) =
            [](Value *v) { return std::to_string(v->m_value); },
        std::string op_name = "Value",
        std::vector<Value *> const &children = {})
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
    ~Value() { std::cerr << "Destructor called" << std::endl; }

    static Value *TempValue(float value) {
        Value *result = new Value(value);
        result->m_requires_grad = false;
        return result;
    }

    std::string to_string() const {
        return m_to_string(const_cast<Value *>(this));
    }
    void backward() {
        m_has_grad = true;
        m_backward_f(this);
    }
    float value() const { return m_value; }
    float grad() const {
        if (!m_has_grad) {
            std::cerr << "grad() called on a node without computed gradient"
                      << std::endl;
        }
        return m_grad;
    }

#define AD_MAKE_TEMP(x) *Value::TempValue(x)
#define AD_BINARY_PERFECT_FORWARD(before, func)                             \
    before Value &func(Value &lhs, Value &&rhs) { return func(lhs, rhs); }  \
    before Value &func(Value &&lhs, Value &rhs) { return func(lhs, rhs); }  \
    before Value &func(Value &&lhs, Value &&rhs) { return func(lhs, rhs); } \
    before Value &func(Value &lhs, float rhs) {                             \
        return func(lhs, AD_MAKE_TEMP(rhs));                                \
    }                                                                       \
    before Value &func(Value &&lhs, float rhs) {                            \
        return func(lhs, AD_MAKE_TEMP(rhs));                                \
    }                                                                       \
    before Value &func(float lhs, Value &rhs) {                             \
        return func(AD_MAKE_TEMP(lhs), rhs);                                \
    }                                                                       \
    before Value &func(float lhs, Value &&rhs) {                            \
        return func(AD_MAKE_TEMP(lhs), rhs);                                \
    }

#define AD_BINARY_OP(op, value, lhs_grad, rhs_grad)                     \
    friend Value &operator op(Value &lhs, Value &rhs) {                 \
        static auto backward_f = [](Value *v) {                         \
            AD_ENSURE_REQUIRES_GRAD(v);                                 \
            auto lhs = v->m_children[0], rhs = v->m_children[1];        \
            if (lhs->m_requires_grad) {                                 \
                lhs->m_grad = lhs_grad;                                 \
                lhs->backward();                                        \
            }                                                           \
            if (rhs->m_requires_grad) {                                 \
                rhs->m_grad = rhs_grad;                                 \
                rhs->backward();                                        \
            }                                                           \
        };                                                              \
        static auto to_string = [](Value *v) {                          \
            return v->m_children[0]->to_string() + #op +                \
                   v->m_children[1]->to_string();                       \
        };                                                              \
        Value *result =                                                 \
            new Value(value, backward_f, to_string, #op, {&lhs, &rhs}); \
        return *result;                                                 \
    }                                                                   \
    AD_BINARY_PERFECT_FORWARD(friend, operator op)

    AD_BINARY_OP(+, lhs.m_value + rhs.m_value, v->m_grad, v->m_grad);
    AD_BINARY_OP(-, lhs.m_value - rhs.m_value, v->m_grad, v->m_grad * -1.0f);
    AD_BINARY_OP(*, lhs.m_value *rhs.m_value, v->m_grad * rhs->m_value,
                 v->m_grad * lhs->m_value);
    AD_BINARY_OP(/, lhs.m_value / rhs.m_value, v->m_grad / rhs->m_value,
                 v->m_grad * lhs->m_value / (rhs->m_value * rhs->m_value));

#define AD_UNARY_OP(op, value, grad)                                          \
    friend Value &operator op(Value &obj) {                                   \
        static auto backward_f = [](Value *v) {                               \
            AD_ENSURE_REQUIRES_GRAD(v);                                       \
            auto obj = v->m_children[0];                                      \
            if (obj->m_requires_grad) {                                       \
                obj->m_grad = grad;                                           \
                obj->backward();                                              \
            }                                                                 \
        };                                                                    \
        static auto to_string = [](Value *v) {                                \
            return #op + v->m_children[0]->to_string();                       \
        };                                                                    \
        Value *result = new Value(value, backward_f, to_string, #op, {&obj}); \
        return *result;                                                       \
    }                                                                         \
    friend Value &operator op(Value &&obj) { return operator op(obj); }

    AD_UNARY_OP(-, -obj.m_value, v->m_grad * -1);

    friend Value &pow(Value &base, Value &exponent);
    friend Value &relu(Value &obj);
};

/// Math functions ///

Value &pow(Value &base, Value &exponent) {
    static auto backward_f = [](Value *v) {
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
    static auto to_string = [](Value *v) {
        return v->m_children[0]->to_string() + "**" +
               v->m_children[1]->to_string();
    };
    Value *result = new Value(std::pow(base.m_value, exponent.m_value),
                              backward_f, to_string, "**", {&base, &exponent});
    return *result;
}
#define EMPTY
AD_BINARY_PERFECT_FORWARD(EMPTY, pow);
#undef EMPTY

/// NN related functions ///

Value &relu(Value &obj) {
    static auto backward_f = [](Value *v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        auto obj = v->m_children[0];
        if (obj->m_requires_grad) {
            obj->m_grad = v->m_value > 0 ? v->m_grad : 0;
            obj->backward();
        }
    };
    static auto to_string = [](Value *v) {
        return "relu(" + v->m_children[0]->to_string() + ")";
    };
    Value *result = new Value(obj.m_value > 0 ? obj.m_value : 0, backward_f,
                              to_string, "relu", {&obj});
    return *result;
}
Value &relu(Value &&v) { return relu(v); }

/// Tensor-like structures ///

};  // namespace ad
