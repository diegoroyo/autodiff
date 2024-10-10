#pragma once

#include "autodiff/autodiff.h"
#include "libcpp-common/geometry.h"
#include "libcpp-common/test.h"

using Value = ad::Value;
using Vector1 = ad::Vector<1>;
using Vector3 = ad::Vector<3>;
using Matrix2 = ad::Matrix<2>;
using Matrix22 = ad::Matrix<2, 2>;
using Matrix23 = ad::Matrix<2, 3>;
COMPILE_TIME_TEST(Value)
COMPILE_TIME_TEST(Vector1)
COMPILE_TIME_TEST(Vector3)
COMPILE_TIME_TEST(Matrix2)
COMPILE_TIME_TEST(Matrix22)
COMPILE_TIME_TEST(Matrix23)

TEST_CASE(00_constructors_Value, {
    TEST_TRUE(Value_compiles_from_type<ad::Value, float>);
    TEST_TRUE(Value_compiles_from_type<ad::Value, int>)
    TEST_TRUE(!Value_compiles_from_type<ad::Value, char*>)
    TEST_TRUE(!Value_compiles_from_type<ad::Value, float, char>)
})

TEST_CASE(01_constructors_Vector, {
    TEST_TRUE(Vector1_compiles_from_type<ad::Vector<1>, float>);
    TEST_TRUE(Vector1_compiles_from_type<ad::Vector<1>, int>)
    TEST_TRUE(Vector1_compiles_from_type<ad::Vector<1>, std::array<float, 1>>)
    TEST_TRUE(Vector1_compiles_from_type<ad::Vector<1>, common::Vec<float, 1>>)
    TEST_TRUE(!Vector1_compiles_from_type<ad::Vector<1>, std::array<float, 3>>)
    TEST_TRUE(!Vector1_compiles_from_type<ad::Vector<1>, common::Vec<float, 3>>)
    TEST_TRUE(!Vector1_compiles_from_type<ad::Vector<1>, char*>)
    TEST_TRUE(!Vector1_compiles_from_type<ad::Vector<1>, float, char>)
})

TEST_CASE(02_constructors_Matrix, {
    TEST_TRUE(Matrix2_compiles_from_type<ad::Matrix<2>, float>);
    TEST_TRUE(Matrix2_compiles_from_type<ad::Matrix<2>, int>)
    // FIXME libcpp-common matrix constructor does not unpack arrays
    // TEST_TRUE(Matrix2_compiles_from_type<
    //           ad::Matrix<2>, std::array<common::Vec<float, 2>, 2>>)
    // TEST_TRUE(Matrix2_compiles_from_type<
    //           ad::Matrix<2>, std::array<float, 4>>)
    TEST_TRUE(Matrix2_compiles_from_type<ad::Matrix<2>, common::Mat<float, 2>>)
    TEST_TRUE(!Matrix2_compiles_from_type<ad::Matrix<2>,
                                          std::array<std::array<float, 3>, 2>>)
    TEST_TRUE(!Matrix2_compiles_from_type<ad::Matrix<2>, common::Mat<float, 3>>)
    TEST_TRUE(!Matrix2_compiles_from_type<ad::Matrix<2>, char*>)
    TEST_TRUE(!Matrix2_compiles_from_type<ad::Matrix<2>, float, char>)
})

TEST_CASE(03_value, {
    ad::Value a(3);
    ad::Value b(2.0f);
    TEST_TRUE(a.value() == 3);
    TEST_TRUE(b.value() == 2.0f)
})

TEST_CASE(04_requires_grad, {
    ad::Value a(3);
    ad::Value b = a + 3;
    TEST_TRUE(a.requires_grad());
    TEST_TRUE(b.requires_grad())
})

TEST_CASE(05_grad_backward, {
    ad::Value a(3);
    ad::Value b = a + 3;
    ad::Value c(3);
    try {
        a.grad();
        TEST_TRUE(false);
    } catch (ad::ADException& e) {
    }
    b.backward();
    TEST_TRUE(a.grad() == 1.0f);
    try {
        c.grad();
        TEST_TRUE(false);
    } catch (ad::ADException& e) {
    }
})

TEST_CASE(06_update, {
    ad::Value a(3);
    ad::Value b = a + 3;
    ad::Value c(3);
    try {
        a.update(1.0f);
        TEST_TRUE(false);
    } catch (ad::ADException& e) {
    }
    b.backward();
    TEST_TRUE(a.grad() == 1.0f);
    a.update(1.0f);
    TEST_TRUE(a.value() == 2.0f);
    TEST_TRUE(b.value() == 6.0f);
    TEST_TRUE(c.value() == 3.0f);
    try {
        c.update(1.0f);
        TEST_TRUE(false);
    } catch (ad::ADException& e) {
    }
    TEST_TRUE(a.value() == 2.0f);
    TEST_TRUE(b.value() == 6.0f);
    TEST_TRUE(c.value() == 3.0f);
})