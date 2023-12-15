import re

import pytest
from tum_prediction.utils_interpolation import Interpolation
from tum_prediction.utils_interpolation import SplineInterpolation

Inp = Interpolation()
epsilon = 1e-6


# Test passed
def test__is_increasing():
    # Test for an empty list
    with pytest.raises(ValueError, match="Points is empty."):
        Inp._is_increasing([])

    # Test for an increasing list
    assert Inp._is_increasing([0.0, 1.5, 3.0, 4.5, 6.0]) is True

    # Test for a list with the same elements
    assert Inp._is_increasing([1.0, 1.0, 1.0]) is False

    # Test for a non-increasing list
    assert Inp._is_increasing([3.0, 2.5, 1.0]) is False

    # Test for a list with one element
    assert Inp._is_increasing([1.0]) is True

    # Test for a list with two elements
    assert Inp._is_increasing([1.0, 2.0]) is True

    # Test for a list with two elements in reverse order
    assert Inp._is_increasing([2.0, 1.0]) is False


# Test passed
def test__is_not_decreasing():
    # Test for an empty list
    with pytest.raises(ValueError, match="Points is empty."):
        Inp._is_not_decreasing([])

    # Test for a non-decreasing list
    assert Inp._is_not_decreasing([0.0, 1.5, 3.0, 4.5, 6.0]) is True

    # Test for a list with the same elements
    assert Inp._is_not_decreasing([1.0, 1.0, 1.0]) is True

    # Test for a decreasing list
    assert Inp._is_not_decreasing([3.0, 2.5, 1.0]) is False

    # Test for a list with one element
    assert Inp._is_not_decreasing([1.0]) is True

    # Test for a list with two elements
    assert Inp._is_not_decreasing([1.0, 2.0]) is True

    # Test for a list with two elements in reverse order
    assert Inp._is_not_decreasing([2.0, 1.0]) is False


# Test passed
def test_validateKeys():
    # Test for an empty base_keys
    with pytest.raises(ValueError, match="Points is empty."):
        Inp.validateKeys([], [1.0, 2.0, 3.0])

    # Test for an empty query_keys
    with pytest.raises(ValueError, match="Points is empty."):
        Inp.validateKeys([0.0, 1.0, 2.0, 3.0], [])

    # Test for base_keys with less than 2 elements
    with pytest.raises(
        ValueError, match=re.escape("The size of points is less than 2. len(base_keys) = 1")
    ):
        Inp.validateKeys([1.0], [1.0, 2.0, 3.0])

    # Test for non-sorted base_keys
    with pytest.raises(ValueError, match="Either base_keys or query_keys is not sorted."):
        Inp.validateKeys([3.0, 1.0, 2.0, 3.0], [0.5, 1.5, 3.0])

    # Test for non-sorted query_keys
    with pytest.raises(ValueError, match="Either base_keys or query_keys is not sorted."):
        Inp.validateKeys([0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0])

    # Test for query_keys out of base_keys range
    with pytest.raises(ValueError, match="query_keys is out of base_keys"):
        Inp.validateKeys([0.0, 1.0, 2.0, 3.0], [-1.0, 1.0, 2.0, 4.0])

    # Test for query_keys slightly out of base_keys range
    validated_query_keys = Inp.validateKeys([0.0, 1.0, 2.0, 3.0], [-0.001, 3.001])
    assert validated_query_keys[0] == pytest.approx(0.0, abs=epsilon)
    assert validated_query_keys[-1] == pytest.approx(3.0, abs=epsilon)

    # Test for valid input
    base_keys = [0.0, 1.0, 2.0, 3.0]
    query_keys = [0.5, 1.5, 3.0]
    assert Inp.validateKeys(base_keys, query_keys) == query_keys


# Test passed
def test_validateKeysAndValues():
    # Test for an empty base_keys
    with pytest.raises(ValueError, match="Points is empty."):
        Inp.validateKeysAndValues([], [1.0, 2.0, 3.0])

    # Test for an empty base_values
    with pytest.raises(ValueError, match="Points is empty."):
        Inp.validateKeysAndValues([0.0, 1.0, 2.0, 3.0], [])

    # Test for base_keys with less than 2 elements
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The size of points is less than 2. len(base_keys) = 1, len(base_values) = 4"
        ),
    ):
        Inp.validateKeysAndValues([1.0], [1.0, 2.0, 3.0, 4.0])

    # Test for base_values with less than 2 elements
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The size of points is less than 2. len(base_keys) = 4, len(base_values) = 1"
        ),
    ):
        Inp.validateKeysAndValues([0.0, 1.0, 2.0, 3.0], [1.0])

    # Test for different sizes of base_keys and base_values
    with pytest.raises(ValueError, match="The size of base_keys and base_values are not the same."):
        Inp.validateKeysAndValues([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    # Test for valid input
    base_keys = [0.0, 1.0, 2.0, 3.0]
    base_values = [0.0, 1.0, 2.0, 3.0]
    Inp.validateKeysAndValues(base_keys, base_values)


# Test passed
def test__easy_lerp():
    assert Inp._easy_lerp(0.0, 1.0, 0.3) == 0.3
    assert Inp._easy_lerp(-0.5, 12.3, 0.3) == 3.34


# Test passed
def test_lerp():
    # straight: query_keys is same as base_keys
    base_keys = [0.0, 1.0, 2.0, 3.0, 4.0]
    base_values = [0.0, 1.5, 3.0, 4.5, 6.0]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.lerp(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # straight: query_keys is random
    base_keys = [0.0, 1.0, 2.0, 3.0, 4.0]
    base_values = [0.0, 1.5, 3.0, 4.5, 6.0]
    query_keys = [0.0, 0.7, 1.9, 4.0]
    ans = [0.0, 1.05, 2.85, 6.0]
    query_values = Inp.lerp(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # curve: query_keys is same as base_keys
    base_keys = [-1.5, 1.0, 5.0, 10.0, 15.0, 20.0]
    base_values = [-1.2, 0.5, 1.0, 1.2, 2.0, 1.0]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.lerp(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # curve: query_keys is same as random
    base_keys = [-1.5, 1.0, 5.0, 10.0, 15.0, 20.0]
    base_values = [-1.2, 0.5, 1.0, 1.2, 2.0, 1.0]
    query_keys = [0.0, 8.0, 18.0]
    ans = [-0.18, 1.12, 1.4]
    query_values = Inp.lerp(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)


# Test passed
def test_splineInterpolation():
    # curve: query_keys is random
    base_keys = [-1.5, 1.0, 5.0, 10.0, 15.0, 20.0]
    base_values = [-1.2, 0.5, 1.0, 1.2, 2.0, 1.0]
    query_keys = [0.0, 8.0, 18.0]
    ans = [-0.075611, 0.997242, 1.573258]

    s = SplineInterpolation(base_keys, base_values)
    query_values = s.getSplineInterpolatedValues(query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)


# Test passed
def test_spline():
    # straight: query_keys is same as base_keys
    base_keys = [0.0, 1.0, 2.0, 3.0, 4.0]
    base_values = [0.0, 1.5, 3.0, 4.5, 6.0]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.spline(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # straight: query_keys is random
    base_keys = [0.0, 1.0, 2.0, 3.0, 4.0]
    base_values = [0.0, 1.5, 3.0, 4.5, 6.0]
    query_keys = [0.0, 0.7, 1.9, 4.0]
    ans = [0.0, 1.05, 2.85, 6.0]
    query_values = Inp.spline(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # curve: query_keys is same as base_keys
    base_keys = [-1.5, 1.0, 5.0, 10.0, 15.0, 20.0]
    base_values = [-1.2, 0.5, 1.0, 1.2, 2.0, 1.0]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.spline(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # curve: query_keys is random
    base_keys = [-1.5, 1.0, 5.0, 10.0, 15.0, 20.0]
    base_values = [-1.2, 0.5, 1.0, 1.2, 2.0, 1.0]
    query_keys = [0.0, 8.0, 18.0]
    ans = [-0.075611, 0.997242, 1.573258]
    query_values = Inp.spline(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # straight: size of base_keys is 2 (edge case in the implementation)
    base_keys = [0.0, 1.0]
    base_values = [0.0, 1.5]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.spline(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # straight: size of base_keys is 3 (edge case in the implementation)
    base_keys = [0.0, 1.0, 2.0]
    base_values = [0.0, 1.5, 3.0]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.spline(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # curve: query_keys is random. size of base_keys is 3 (edge case in the implementation)
    base_keys = [-1.5, 1.0, 5.0]
    base_values = [-1.2, 0.5, 1.0]
    query_keys = [-1.0, 0.0, 4.0]
    ans = [-0.808769, -0.077539, 1.035096]
    query_values = Inp.spline(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # When the query keys changes suddenly (edge case of spline interpolation).
    base_keys = [0.0, 1.0, 1.0001, 2.0, 3.0, 4.0]
    base_values = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1]
    query_keys = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]
    ans = [0.0, 0.0, 137.591789, 0.1, 0.1, 0.1]
    query_values = Inp.spline(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)


# Test passed
def test_spline_by_akima():
    # straight: query_keys is same as base_keys
    base_keys = [0.0, 1.0, 2.0, 3.0, 4.0]
    base_values = [0.0, 1.5, 3.0, 4.5, 6.0]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.spline_by_akima(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # straight: query_keys is random
    base_keys = [0.0, 1.0, 2.0, 3.0, 4.0]
    base_values = [0.0, 1.5, 3.0, 4.5, 6.0]
    query_keys = [0.0, 0.7, 1.9, 4.0]
    ans = [0.0, 1.05, 2.85, 6.0]
    query_values = Inp.spline_by_akima(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # curve: query_keys is same as base_keys
    base_keys = [-1.5, 1.0, 5.0, 10.0, 15.0, 20.0]
    base_values = [-1.2, 0.5, 1.0, 1.2, 2.0, 1.0]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.spline_by_akima(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # curve: query_keys is random
    base_keys = [-1.5, 1.0, 5.0, 10.0, 15.0, 20.0]
    base_values = [-1.2, 0.5, 1.0, 1.2, 2.0, 1.0]
    query_keys = [0.0, 8.0, 18.0]
    ans = [-0.0801, 1.110749, 1.4864]
    query_values = Inp.spline_by_akima(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # straight: size of base_keys is 2 (edge case in the implementation)
    base_keys = [0.0, 1.0]
    base_values = [0.0, 1.5]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.spline_by_akima(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # straight: size of base_keys is 3 (edge case in the implementation)
    base_keys = [0.0, 1.0, 2.0]
    base_values = [0.0, 1.5, 3.0]
    query_keys = base_keys
    ans = base_values
    query_values = Inp.spline_by_akima(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # curve: query_keys is random. size of base_keys is 3 (edge case in the implementation)
    base_keys = [-1.5, 1.0, 5.0]
    base_values = [-1.2, 0.5, 1.0]
    query_keys = [-1.0, 0.0, 4.0]
    ans = [-0.8378, -0.0801, 0.927031]
    query_values = Inp.spline_by_akima(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)

    # When the query keys changes suddenly (edge case of spline interpolation).
    base_keys = [0.0, 1.0, 1.0001, 2.0, 3.0, 4.0]
    base_values = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1]
    query_keys = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]
    ans = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1]
    query_values = Inp.spline_by_akima(base_keys, base_values, query_keys)
    for i in range(len(query_keys)):
        assert query_values[i] == pytest.approx(ans[i], abs=epsilon)
