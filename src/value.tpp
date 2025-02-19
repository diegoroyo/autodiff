namespace ad {

/// Math functions ///

#define AD_POW_TEMPLATE               \
    template <typename B, typename E, \
              typename = std::enable_if_t<std::is_scalar_v<E>>>
AD_POW_TEMPLATE
_ValueWrapper<B> pow(_ValueWrapper<B>& base, _ValueWrapper<E>& exponent) {
    static auto backward_f = [](_ValueData<B>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<B>& base = v.template get_child<B>(0);
        _ValueData<E>& exponent = v.template get_child<E>(1);
        if (base.m_requires_grad) {
            // d/dx x^n = n*x^(n-1)
            base.m_grad = ad::detail::ewise_mult(
                v.m_grad * exponent.m_value,
                ad::detail::pow(base.m_value, exponent.m_value - 1.0f));
            base.backward();
        }
        if (exponent.m_requires_grad) {
            throw ADException(
                "NYI. gradient cannot be computed for the base in ad::pow");
            // NOTE this could work, but needs to be tested
            // d/dx n^x = n^x log(n)
            // exponent->m_grad =
            //   ad::detail::sum(ad::detail::ewise_mult(
            //     v->m_grad, v->m_value * std::log(base->m_value)));
            // exponent.backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _ValueData<B>& v) -> std::ostream& {
        _ValueData<B>& base = v.template get_child<B>(0);
        _ValueData<E>& exponent = v.template get_child<E>(1);
        o << base << "**" << exponent;
        return o;
    };
    _ValueWrapper<B> result(ad::detail::pow(base.value(), exponent.value()),
                            backward_f, to_string, "**",
                            {AD_CHILD(base), AD_CHILD(exponent)});
    base.set_parent(result);
    exponent.set_parent(result);
    return result;
}
AD_POW_TEMPLATE
_ValueWrapper<B> pow(_ValueWrapper<B>& lhs, _ValueWrapper<E>&& rhs) {
    return pow(lhs, rhs);
}
AD_POW_TEMPLATE
_ValueWrapper<B> pow(_ValueWrapper<B>&& lhs, _ValueWrapper<E>& rhs) {
    return pow(lhs, rhs);
}
AD_POW_TEMPLATE
_ValueWrapper<B> pow(_ValueWrapper<B>&& lhs, _ValueWrapper<E>&& rhs) {
    return pow(lhs, rhs);
}
AD_POW_TEMPLATE
_ValueWrapper<B> pow(_ValueWrapper<B>& lhs, E rhs) {
    return pow(lhs, AD_MAKE_TEMP(rhs, E));
}
AD_POW_TEMPLATE
_ValueWrapper<B> pow(_ValueWrapper<B>&& lhs, E rhs) {
    return pow(lhs, AD_MAKE_TEMP(rhs, E));
}
AD_POW_TEMPLATE
_ValueWrapper<B> pow(B lhs, _ValueWrapper<E>& rhs) {
    return pow(AD_MAKE_TEMP(lhs, B), rhs);
}
AD_POW_TEMPLATE
_ValueWrapper<B> pow(B lhs, _ValueWrapper<E>&& rhs) {
    return pow(AD_MAKE_TEMP(lhs, B), rhs);
}
#undef AD_POW_TEMPLATE

#define AD_LOG_TEMPLATE               \
    template <typename V, typename B, \
              typename = std::enable_if_t<std::is_scalar_v<B>>>
AD_LOG_TEMPLATE
_ValueWrapper<V> log(_ValueWrapper<V>& value, _ValueWrapper<B>& base) {
    static auto backward_f = [](_ValueData<V>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<V>& value = v.template get_child<V>(0);
        _ValueData<B>& base = v.template get_child<B>(1);
        if (value.m_requires_grad) {
            // d/dx log_b(x) = 1/(x * log(b))
            value.m_grad = v.m_grad / (value.m_value * std::log(base.m_value));
            value.backward();
        }
        if (base.m_requires_grad) {
            throw ADException(
                "NYI. gradient cannot be computed for the base in ad::log");
            base.backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _ValueData<V>& v) -> std::ostream& {
        _ValueData<V>& value = v.template get_child<V>(0);
        _ValueData<B>& base = v.template get_child<B>(1);
        o << "log_[" << base << "] (" << value << ")";
        return o;
    };
    _ValueWrapper<V> result(ad::detail::log(value.value(), base.value()),
                            backward_f, to_string, "log",
                            {AD_CHILD(value), AD_CHILD(base)});
    value.set_parent(result);
    base.set_parent(result);
    return result;
}
AD_LOG_TEMPLATE
_ValueWrapper<V> log(_ValueWrapper<V>& lhs, _ValueWrapper<B>&& rhs) {
    return log(lhs, rhs);
}
AD_LOG_TEMPLATE
_ValueWrapper<V> log(_ValueWrapper<V>&& lhs, _ValueWrapper<B>& rhs) {
    return log(lhs, rhs);
}
AD_LOG_TEMPLATE
_ValueWrapper<V> log(_ValueWrapper<V>&& lhs, _ValueWrapper<B>&& rhs) {
    return log(lhs, rhs);
}
AD_LOG_TEMPLATE
_ValueWrapper<V> log(_ValueWrapper<V>& lhs, B rhs) {
    return log(lhs, AD_MAKE_TEMP(rhs, B));
}
AD_LOG_TEMPLATE
_ValueWrapper<V> log(_ValueWrapper<V>&& lhs, B rhs) {
    return log(lhs, AD_MAKE_TEMP(rhs, B));
}
template <typename V>
_ValueWrapper<V> log(_ValueWrapper<V>& lhs, float rhs = std::exp(1)) {
    return log(lhs, AD_MAKE_TEMP(rhs, float));
}
template <typename V>
_ValueWrapper<V> log(_ValueWrapper<V>&& lhs, float rhs = std::exp(1)) {
    return log(lhs, AD_MAKE_TEMP(rhs, float));
}
AD_LOG_TEMPLATE
_ValueWrapper<V> log(V lhs, _ValueWrapper<B>& rhs) {
    return log(AD_MAKE_TEMP(lhs, B), rhs);
}
AD_LOG_TEMPLATE
_ValueWrapper<V> log(V lhs, _ValueWrapper<B>&& rhs) {
    return log(AD_MAKE_TEMP(lhs, B), rhs);
}
#undef AD_LOG_TEMPLATE

/// Element-wise operations ///

template <typename T>
_ValueWrapper<T> exp(_ValueWrapper<T>& obj) {
    static auto backward_f = [](_ValueData<T>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<T>& child = v.template get_child<T>(0);
        if (child.m_requires_grad) {
            child.m_grad = ad::detail::ewise_mult(
                ad::detail::ewise_mult(child.m_value, v.m_value), v.m_grad);
            child.backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _ValueData<T>& v) -> std::ostream& {
        o << "exp(" << v.template get_child<T>(0) << ")";
        return o;
    };
    _ValueWrapper<T> result(ad::detail::exp(obj.value()), backward_f, to_string,
                            "exp", {AD_CHILD(obj)});
    obj.set_parent(result);
    return result;
}
template <typename T>
_ValueWrapper<T> exp(_ValueWrapper<T>&& v) {
    return exp(v);
}

template <typename T>
_ValueWrapper<T> relu(_ValueWrapper<T>& obj) {
    static auto backward_f = [](_ValueData<T>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<T>& child = v.template get_child<T>(0);
        if (child.m_requires_grad) {
            child.m_grad = ad::detail::relu_helper(v.m_value, v.m_grad);
            child.backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _ValueData<T>& v) -> std::ostream& {
        o << "relu(" << v.template get_child<T>(0) << ")";
        return o;
    };
    _ValueWrapper<T> result(ad::detail::relu_helper(obj.value(), obj.value()),
                            backward_f, to_string, "relu", {AD_CHILD(obj)});
    obj.set_parent(result);
    return result;
}
template <typename T>
_ValueWrapper<T> relu(_ValueWrapper<T>&& v) {
    return relu(v);
}

template <typename T>
_ValueWrapper<T> sigmoid(_ValueWrapper<T>& obj) {
    static auto backward_f = [](_ValueData<T>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<T>& child = v.template get_child<T>(0);
        if (child.m_requires_grad) {
            child.m_grad = ad::detail::ewise_mult(
                v.m_grad, ad::detail::ewise_mult(v.m_value, -v.m_value + 1));
            child.backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _ValueData<T>& v) -> std::ostream& {
        o << "sigmoid(" << v.template get_child<T>(0) << ")";
        return o;
    };
    _ValueWrapper<T> result(ad::detail::sigmoid(obj.value()), backward_f,
                            to_string, "sigmoid", {AD_CHILD(obj)});
    obj.set_parent(result);
    return result;
}
template <typename T>
_ValueWrapper<T> sigmoid(_ValueWrapper<T>&& v) {
    return sigmoid(v);
}

template <typename T>
_ValueWrapper<T> sin(_ValueWrapper<T>& obj) {
    static auto backward_f = [](_ValueData<T>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<T>& child = v.template get_child<T>(0);
        if (child.m_requires_grad) {
            child.m_grad = v.m_grad * ad::detail::cos(child.m_value);
            child.backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _ValueData<T>& v) -> std::ostream& {
        o << "sin(" << v.template get_child<T>(0) << ")";
        return o;
    };
    _ValueWrapper<T> result(ad::detail::sin(obj.value()), backward_f, to_string,
                            "sin", {AD_CHILD(obj)});
    obj.set_parent(result);
    return result;
}
template <typename T>
_ValueWrapper<T> sin(_ValueWrapper<T>&& v) {
    return sin(v);
}

template <typename T>
_ValueWrapper<T> cos(_ValueWrapper<T>& obj) {
    static auto backward_f = [](_ValueData<T>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<T>& child = v.template get_child<T>(0);
        if (child.m_requires_grad) {
            child.m_grad = v.m_grad * ad::detail::sin(child.m_value) * -1.0f;
            child.backward();
        }
    };
    static auto to_string = [](std::ostream& o,
                               const _ValueData<T>& v) -> std::ostream& {
        o << "cos(" << v.template get_child<T>(0) << ")";
        return o;
    };
    _ValueWrapper<T> result(ad::detail::cos(obj.value()), backward_f, to_string,
                            "cos", {AD_CHILD(obj)});
    obj.set_parent(result);
    return result;
}
template <typename T>
_ValueWrapper<T> cos(_ValueWrapper<T>&& v) {
    return cos(v);
}

/// Tensor reduce operations ///

template <typename T>
Value sum(_ValueWrapper<T>& obj) {
    static auto backward_f = [](_ValueData<typename Value::type>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<T>& child = v.template get_child<T>(0);
        if (child.m_requires_grad) {
            child.m_grad = v.m_grad;
            child.backward();
        }
    };
    static auto to_string =
        [](std::ostream& o,
           const _ValueData<typename Value::type>& v) -> std::ostream& {
        o << "sum(" << v.template get_child<T>(0) << ")";
        return o;
    };
    Value result(detail::sum(obj.value()), backward_f, to_string, "sum",
                 {AD_CHILD(obj)});
    obj.set_parent(result);
    return result;
}
template <typename T>
Value sum(_ValueWrapper<T>&& v) {
    return sum(v);
}

/// Tensor expand operations ///

template <size_t N>
Vector<N> expand(Value& obj) {
    static auto backward_f = [](_ValueData<typename Vector<N>::type>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<typename Value::type>& child =
            v.template get_child<typename Value::type>(0);
        if (child.m_requires_grad) {
            child.m_grad = detail::sum(v.m_grad);
            child.backward();
        }
    };
    static auto to_string =
        [](std::ostream& o,
           const _ValueData<typename Vector<N>::type>& v) -> std::ostream& {
        for (size_t i = 0; i < N; ++i)
            o << v.template get_child<typename Value::type>(0).m_value;
        return o;
    };
    Vector<N> result(obj.value(), backward_f, to_string,
                     "expand(" + std::to_string(N) + ")", {AD_CHILD(obj)});
    obj.set_parent(result);
    return result;
}
template <size_t N>
Vector<N> expand(Value&& obj) {
    return expand<N>(obj);
}

template <size_t N, size_t S>
Vector<S * N> expand(Vector<S>& obj) {
    static auto backward_f = [](_ValueData<typename Vector<S * N>::type>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<typename Vector<S>::type>& child =
            v.template get_child<typename Vector<S>::type>(0);
        if (child.m_requires_grad) {
            child.m_grad = 0;
            for (size_t i = 0; i < S; ++i)
                for (size_t j = i; j < S * N; j += N)
                    child.m_grad(i) += v.m_grad(j);
            child.backward();
        }
    };
    static auto to_string =
        [](std::ostream& o,
           const _ValueData<typename Vector<S * N>::type>& v) -> std::ostream& {
        for (size_t i = 0; i < N; ++i)
            o << v.template get_child<typename Vector<S>::type>(0).m_value;
        return o;
    };
    Vector<S * N> result({}, backward_f, to_string,
                         "expand(" + std::to_string(N) + ")", {AD_CHILD(obj)});
    obj.set_parent(result);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < S; ++j)
            result.value()(i * S + j) = obj.value()(j);
    return result;
}
template <size_t N, size_t S>
Vector<S * N> expand(Vector<S>&& obj) {
    return expand<N, S>(obj);
}

template <size_t... Shape>
Vector<(Shape * ...)> flatten(Tensor<Shape...>& obj) {
    constexpr size_t size = (Shape * ...);
    static auto backward_f = [](_ValueData<typename Vector<size>::type>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<typename Tensor<Shape...>::type>& child =
            v.template get_child<typename Tensor<Shape...>::type>(0);
        if (child.m_requires_grad) {
            for (size_t i = 0; i < size; ++i) child.m_grad.at(i) = v.m_grad(i);
            child.backward();
        }
    };
    static auto to_string =
        [](std::ostream& o,
           const _ValueData<typename Vector<size>::type>& v) -> std::ostream& {
        o << v.template get_child<typename Tensor<Shape...>::type>(0).m_value;
        return o;
    };
    Vector<size> result({}, backward_f, to_string, "flatten", {AD_CHILD(obj)});
    obj.set_parent(result);
    for (size_t i = 0; i < size; ++i) result.value()(i) = obj.value().at(i);
    return result;
}
template <size_t... Shape>
Vector<Tensor<Shape...>::type::size> flatten(Tensor<Shape...>&& obj) {
    return flatten(obj);
}

};  // namespace ad