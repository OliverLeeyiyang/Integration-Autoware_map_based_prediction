# solve Ax = d
# where A is tridiagonal matrix
#     [b_0 c_0 ...                       ]
#     [a_0 b_1 c_1 ...               O   ]
# A = [            ...                   ]
#     [   O         ... a_N-3 b_N-2 c_N-2]
#     [                   ... a_N-2 b_N-1]

import numpy as np


class TDMACoef:
    def __init__(self, num_row):
        self.a = np.zeros(num_row - 1, dtype=float)
        self.b = np.zeros(num_row, dtype=float)
        self.c = np.zeros(num_row - 1, dtype=float)
        self.d = np.zeros(num_row, dtype=float)

class MultiSplineCoef:
    # NOTE: X(s) = a_i (s - s_i)^3 + b_i (s - s_i)^2 + c_i (s - s_i) + d_i : (i = 0, 1, ... N-1)
    def __init__(self, num_spline):
        self.a = np.zeros(num_spline, dtype=float)
        self.b = np.zeros(num_spline, dtype=float)
        self.c = np.zeros(num_spline, dtype=float)
        self.d = np.zeros(num_spline, dtype=float)

def solveTridiagonalMatrixAlgorithm(tdma_coef: TDMACoef) -> np.ndarray:
    a = tdma_coef.a
    b = tdma_coef.b
    c = tdma_coef.c
    d = tdma_coef.d

    num_row = len(b)

    x = np.zeros(num_row, dtype=float)

    if num_row != 1:
        # calculate p and q
        p = np.zeros(num_row, dtype=float)
        q = np.zeros(num_row, dtype=float)

        p[0] = -c[0] / b[0]
        q[0] = d[0] / b[0]

        for i in range(1, num_row):
            den = b[i] + a[i - 1] * p[i - 1]
            p[i] = -c[i - 1] / den
            q[i] = (d[i] - a[i - 1] * q[i - 1]) / den

        # calculate solution
        x[num_row - 1] = q[num_row - 1]

        for i in range(1, num_row):
            j = num_row - 1 - i
            x[j] = p[j] * x[j + 1] + q[j]
    else:
        x[0] = d[0] / b[0]

    return x

def check_data_type(data):
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError("data must be list or numpy.ndarray.")

    return data

class Interpolation:
    def __init__(self):
        pass

    def _is_increasing(self, x: np.ndarray) -> bool:
        if len(x) == 0:
            raise ValueError("Points is empty.")

        return np.all(x[:-1] < x[1:])

    def _is_not_decreasing(self, x: np.ndarray) -> bool:
        if len(x) == 0:
            raise ValueError("Points is empty.")

        return np.all(x[:-1] <= x[1:])

    def validateKeys(self, base_keys: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
        # When vectors are empty
        if len(base_keys) == 0 or len(query_keys) == 0:
            raise ValueError("Points is empty.")

        # When size of vectors is less than 2
        if len(base_keys) < 2:
            raise ValueError(
                "The size of points is less than 2. len(base_keys) = " + str(len(base_keys))
            )

        # When indices are not sorted
        if not self._is_increasing(base_keys) or not self._is_not_decreasing(query_keys):
            raise ValueError("Either base_keys or query_keys is not sorted.")

        # When query_keys are out of base_keys (This function does not allow exterior division.)
        epsilon = 1e-3
        if query_keys[0] < base_keys[0] - epsilon or query_keys[-1] > base_keys[-1] + epsilon:
            raise ValueError("query_keys is out of base_keys")

        # NOTE: Due to calculation error of float, a query key may be slightly out of base keys.
        #       Therefore, query keys are cropped here.
        validated_query_keys = query_keys.copy()
        validated_query_keys[0] = max(validated_query_keys[0], base_keys[0])
        validated_query_keys[-1] = min(validated_query_keys[-1], base_keys[-1])

        return validated_query_keys

    def validateKeysAndValues(self, base_keys: np.ndarray, base_values: np.ndarray):
        # When vectors are empty
        if len(base_keys) == 0 or len(base_values) == 0:
            raise ValueError("Points is empty.")

        # When size of vectors is less than 2
        if len(base_keys) < 2 or len(base_values) < 2:
            raise ValueError(
                "The size of points is less than 2. len(base_keys) = "
                + str(len(base_keys))
                + ", len(base_values) = "
                + str(len(base_values))
            )

        # When sizes of indices and values are not the same
        if len(base_keys) != len(base_values):
            raise ValueError("The size of base_keys and base_values are not the same.")

    def _easy_lerp(self, src_val: float, dst_val: float, ratio: float) -> float:
        return src_val + (dst_val - src_val) * ratio

    def lerp(
        self, base_keys: np.ndarray, base_values: np.ndarray, query_keys: np.ndarray
    ) -> np.ndarray:
        base_keys = check_data_type(base_keys)
        base_values = check_data_type(base_values)
        query_keys = check_data_type(query_keys)

        # Throw exception for invalid arguments
        validated_query_keys = self.validateKeys(base_keys, query_keys)
        self.validateKeysAndValues(base_keys, base_values)

        # Calculate linear interpolation
        query_values = np.zeros(len(validated_query_keys), dtype=float)
        key_index = 0
        for i, query_key in enumerate(validated_query_keys):
            while base_keys[key_index + 1] < query_key:
                key_index += 1

            src_val = base_values[key_index]
            dst_val = base_values[key_index + 1]
            ratio = (query_key - base_keys[key_index]) / (
                base_keys[key_index + 1] - base_keys[key_index]
            )

            interpolated_val = self._easy_lerp(src_val, dst_val, ratio)
            query_values[i] = interpolated_val

        return query_values

    def spline(
        self, base_keys: np.ndarray, base_values: np.ndarray, query_keys: np.ndarray
    ) -> np.ndarray:
        base_keys = check_data_type(base_keys)
        base_values = check_data_type(base_values)
        query_keys = check_data_type(query_keys)

        # Calculate spline coefficients
        interpolator = SplineInterpolation(base_keys, base_values)

        # Interpolate base_keys at query_keys
        return interpolator.getSplineInterpolatedValues(query_keys)

    def spline_by_akima(
        self, base_keys: np.ndarray, base_values: np.ndarray, query_keys: np.ndarray
    ) -> np.ndarray:
        base_keys = check_data_type(base_keys)
        base_values = check_data_type(base_values)
        query_keys = check_data_type(query_keys)

        epsilon = 1e-5

        # calculate m
        m_values = np.zeros(len(base_keys) - 1, dtype=float)
        for i in range(len(base_keys) - 1):
            m_values[i] = (base_values[i + 1] - base_values[i]) / (base_keys[i + 1] - base_keys[i])

        # calculate s
        s_values = np.zeros(len(base_keys), dtype=float)
        for i in range(len(base_keys)):
            if i == 0:
                s_values[i] = m_values[0]
                continue
            elif i == len(base_keys) - 1:
                s_values[i] = m_values[-1]
                continue
            elif i == 1 or i == len(base_keys) - 2:
                s_val = (m_values[i - 1] + m_values[i]) / 2.0
                s_values[i] = s_val
                continue

            denom = abs(m_values[i + 1] - m_values[i]) + abs(m_values[i - 1] - m_values[i - 2])
            if abs(denom) < epsilon:
                s_val = (m_values[i - 1] + m_values[i]) / 2.0
                s_values[i] = s_val
                continue

            s_val = (
                abs(m_values[i + 1] - m_values[i]) * m_values[i - 1]
                + abs(m_values[i - 1] - m_values[i - 2]) * m_values[i]
            ) / denom
            s_values[i] = s_val

        # calculate cubic coefficients
        a = np.zeros(len(base_keys) - 1, dtype=float)
        b = np.zeros(len(base_keys) - 1, dtype=float)
        c = s_values[:-1].copy()
        d = base_values[:-1].copy()

        for i in range(len(base_keys) - 1):
            a[i] = (s_values[i] + s_values[i + 1] - 2.0 * m_values[i]) / (base_keys[i + 1] - base_keys[i]) ** 2
            b[i] = (3.0 * m_values[i] - 2.0 * s_values[i] - s_values[i + 1]) / (base_keys[i + 1] - base_keys[i])

        # interpolate
        res = np.zeros(len(query_keys), dtype=float)
        j = 0
        for k, query_key in enumerate(query_keys):
            while base_keys[j + 1] < query_key:
                j += 1

            ds = query_key - base_keys[j]
            interpolated_val = d[j] + (c[j] + (b[j] + a[j] * ds) * ds) * ds
            res[k] = interpolated_val

        return res

class SplineInterpolation:
    def __init__(self, base_keys: np.ndarray, base_values: np.ndarray):
        self.base_keys_ = None
        self.multi_spline_coef_ = None

        self.interpolation = Interpolation()

        base_keys = check_data_type(base_keys)
        base_values = check_data_type(base_values)
        self.calcSplineCoefficients(base_keys, base_values)

    def calcSplineCoefficients(self, base_keys: np.ndarray, base_values: np.ndarray):
        # throw exceptions for invalid arguments
        self.interpolation.validateKeysAndValues(base_keys, base_values)

        num_base = len(base_keys)  # N+1

        diff_keys = base_keys[1:] - base_keys[:-1]  # N
        diff_values = base_values[1:] - base_values[:-1]  # N

        v = np.zeros(num_base, dtype=float)
        if num_base > 2:
            # solve tridiagonal matrix algorithm
            tdma_coef = TDMACoef(num_base - 2)  # N-1

            tdma_coef.b = 2 * (diff_keys[:-1] + diff_keys[1:])
            tdma_coef.a = diff_keys[1:]
            tdma_coef.c = diff_keys[:-1]
            tdma_coef.d = 6.0 * (diff_values[1:] / diff_keys[1:] - diff_values[:-1] / diff_keys[:-1])

            tdma_res = solveTridiagonalMatrixAlgorithm(tdma_coef)

            # calculate v
            v[1:-1] = tdma_res

        # calculate a, b, c, d of spline coefficients
        self.multi_spline_coef_ = MultiSplineCoef(num_base - 1)  # N

        self.multi_spline_coef_.a = (v[1:] - v[:-1]) / 6.0 / diff_keys
        self.multi_spline_coef_.b = v[:-1] / 2.0
        self.multi_spline_coef_.c = (
            diff_values / diff_keys - diff_keys * (2 * v[:-1] + v[1:]) / 6.0
        )
        self.multi_spline_coef_.d = base_values[:-1]

        self.base_keys_ = base_keys

    def getSplineInterpolatedValues(self, query_keys: np.ndarray) -> np.ndarray:
        # throw exceptions for invalid arguments
        validated_query_keys = self.interpolation.validateKeys(self.base_keys_, query_keys)

        a = self.multi_spline_coef_.a
        b = self.multi_spline_coef_.b
        c = self.multi_spline_coef_.c
        d = self.multi_spline_coef_.d

        res = np.zeros(len(validated_query_keys), dtype=float)
        j = 0
        for k, query_key in enumerate(validated_query_keys):
            while self.base_keys_[j + 1] < query_key:
                j += 1

            ds = query_key - self.base_keys_[j]
            res[k] = d[j] + (c[j] + (b[j] + a[j] * ds) * ds) * ds

        return res

    def getSplineInterpolatedDiffValues(self, query_keys: np.ndarray) -> np.ndarray:
        validated_query_keys = self.interpolation.validateKeys(self.base_keys_, query_keys)

        a = self.multi_spline_coef_.a
        b = self.multi_spline_coef_.b
        c = self.multi_spline_coef_.c

        res = np.zeros(len(validated_query_keys), dtype=float)
        j = 0
        for k, query_key in enumerate(validated_query_keys):
            while self.base_keys_[j + 1] < query_key:
                j += 1

            ds = query_key - self.base_keys_[j]
            res[k] = c[j] + (2.0 * b[j] + 3.0 * a[j] * ds) * ds

        return res

    def getSplineInterpolatedQuadDiffValues(self, query_keys: np.ndarray) -> np.ndarray:
        validated_query_keys = self.interpolation.validateKeys(self.base_keys_, query_keys)

        a = self.multi_spline_coef_.a
        b = self.multi_spline_coef_.b

        res = np.zeros(len(validated_query_keys), dtype=float)
        j = 0
        for k, query_key in enumerate(validated_query_keys):
            while self.base_keys_[j + 1] < query_key:
                j += 1

            ds = query_key - self.base_keys_[j]
            res[k] = 2.0 * b[j] + 6.0 * a[j] * ds

        return res
