namespace ad {

/// Math functions ///

#define AD_POW_TEMPLATE                                             \
    template <typename B, typename E,                               \
              typename = std::enable_if_t<!detail::is_value_v<B> && \
                                          std::is_scalar_v<E>>>
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
            // NOTE this would just work if we implemented element-wise log for
            // vector/matrix types. But it's too much work for sth that is not
            // going to be used
            // d/dx n^x = n^x log(n) exponent->m_grad =
            // ad::detail::sum(ad::detail::ewise_mult(
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

/// Element-wise operations ///

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
Value sum(Value&& v) {
    return sum(v);
}

/// Tensor expand operations ///

template <unsigned int N>
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
        for (unsigned int i = 0; i < N; ++i)
            o << v.template get_child<typename Value::type>(0).m_value;
        return o;
    };
    Vector<N> result(obj.value(), backward_f, to_string,
                     "expand(" + std::to_string(N) + ")", {AD_CHILD(obj)});
    obj.set_parent(result);
    return result;
}
template <unsigned int N>
Vector<N> expand(Value&& obj) {
    return expand<N>(obj);
}

template <unsigned int N, unsigned int S>
Vector<S * N> expand(Vector<S>& obj) {
    static auto backward_f = [](_ValueData<typename Vector<S * N>::type>& v) {
        AD_ENSURE_REQUIRES_GRAD(v);
        _ValueData<typename Vector<S>::type>& child =
            v.template get_child<typename Vector<S>::type>(0);
        if (child.m_requires_grad) {
            child.m_grad = 0;
            for (unsigned int i = 0; i < S; ++i)
                for (unsigned int j = i; j < S * N; j += N)
                    child.m_grad[i] += v.m_grad[j];
            child.backward();
        }
    };
    static auto to_string =
        [](std::ostream& o,
           const _ValueData<typename Vector<S * N>::type>& v) -> std::ostream& {
        for (unsigned int i = 0; i < N; ++i)
            o << v.template get_child<typename Vector<S>::type>(0).m_value;
        return o;
    };
    Vector<S * N> result({}, backward_f, to_string,
                         "expand(" + std::to_string(N) + ")", {AD_CHILD(obj)});
    obj.set_parent(result);
    for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < S; ++j)
            result.value()[i * S + j] = obj.value()[j];
    return result;
}
template <unsigned int N, unsigned int S>
Vector<S * N> expand(Vector<S>&& obj) {
    return expand<N, S>(obj);
}

};  // namespace ad