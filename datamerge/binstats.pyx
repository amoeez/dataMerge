# cython: infer_types=True


cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport sqrt

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sum_weights(np.ndarray[double, ndim=1] weights):
    cdef double sum_weights = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    cdef double[::1] view_w = weights
    for i in range(N):
        sum_weights += view_w[i]
    return sum_weights

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sum(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] weights):
    cdef double sum = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    cdef double[::1] view_v = values
    cdef double[::1] view_w = weights
    for i in range(N):
        sum += view_v[i]* view_w[i]
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sum_sp(np.ndarray[float, ndim=1] values, np.ndarray[double, ndim=1] weights):
    cdef double sum = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    cdef float[::1] view_v = values
    cdef double[::1] view_w = weights
    for i in range(N):
        sum += view_v[i]* view_w[i]
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double weighted_mean(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] weights):
    return sum(values, weights) / sum_weights(weights)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double weighted_mean_sp(np.ndarray[float, ndim=1] values, np.ndarray[double, ndim=1] weights):
    return sum_sp(values, weights) / sum_weights(weights)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[::1] demeaned(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] weights):
    cdef double w_mean = weighted_mean(values, weights)
    cdef Py_ssize_t N = weights.shape[0]
    cdef double[::1] view_v = values
    for i in range(N):
        view_v[i] =  view_v[i] - w_mean
    return values

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[::1] demeaned_sp(np.ndarray[float, ndim=1] values, np.ndarray[double, ndim=1] weights):
    cdef double w_mean = weighted_mean_sp(values, weights)
    cdef Py_ssize_t N = weights.shape[0]
    cdef float[::1] view_v = values
    for i in range(N):
        view_v[i] =  view_v[i] - w_mean
    return values

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sumsquares(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] weights):
    cdef double[::1] view_demeaned = demeaned(values, weights)
    cdef double[::1] view_w = weights
    cdef double sum = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    for i in range(N):
        sum+=  view_demeaned[i]* view_demeaned[i]* view_w[i]
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sumsquares_sp(np.ndarray[float, ndim=1] values, np.ndarray[double, ndim=1] weights):
    cdef float[::1] view_demeaned = demeaned_sp(values, weights)
    cdef double[::1] view_w = weights
    cdef double sum = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    for i in range(N):
        sum+=  view_demeaned[i]* view_demeaned[i]* view_w[i]
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sigma(np.ndarray[double, ndim=1] sigma_values, np.ndarray[double, ndim=1] weights):
    cdef double[::1] view_sigma = sigma_values
    cdef double[::1] view_w = weights
    cdef double sum = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    for i in range(N):
        sum+=  view_sigma[i]* view_sigma[i]* view_w[i]* view_w[i]
    return sqrt(sum)/sum_weights(weights)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sigma_sp(np.ndarray[float, ndim=1] sigma_values, np.ndarray[double, ndim=1] weights):
    cdef float[::1] view_sigma = sigma_values
    cdef double[::1] view_w = weights
    cdef double sum = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    for i in range(N):
        sum+=  view_sigma[i]* view_sigma[i]* view_w[i]* view_w[i]
    return sqrt(sum)/sum_weights(weights)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double var_ddof(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] weights, int ddof =0):
    return sumsquares(values, weights)/(sum_weights(weights) - ddof)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double var_ddof_sp(np.ndarray[float, ndim=1] values, np.ndarray[double, ndim=1] weights, int ddof =0):
    return sumsquares_sp(values, weights)/(sum_weights(weights) - ddof)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double std_ddof(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] weights, int ddof =0):
    return sqrt(var_ddof(values, weights, ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double std_ddof_sp(np.ndarray[float, ndim=1] values, np.ndarray[double, ndim=1] weights, int ddof =0):
    return sqrt(var_ddof_sp(values, weights, ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sem(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] weights, int ddof = 0):
    cdef double weights_sum = sum_weights(weights)
    cdef double sum_squared_weights = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    cdef double[::1] view_w = weights
    for i in range(N):
        sum_squared_weights += view_w[i] * view_w[i]
    return std_ddof(values, weights, ddof) * sqrt(sum_squared_weights / (weights_sum*weights_sum))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sem_sp(np.ndarray[float, ndim=1] values, np.ndarray[double, ndim=1] weights, int ddof = 0):
    cdef double weights_sum = sum_weights(weights)
    cdef double sum_squared_weights = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t N = weights.shape[0]
    cdef double[::1] view_w = weights
    for i in range(N):
        sum_squared_weights += view_w[i] * view_w[i]
    return std_ddof_sp(values, weights, ddof) * sqrt(sum_squared_weights / (weights_sum*weights_sum))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double propagated_error( double sem_value, double sigma_value, double mean_value, double min_multiplier):
    if sem_value >= sigma_value and sem_value >= mean_value*min_multiplier:
        return sem_value
    elif sigma_value >= sem_value and sigma_value >= mean_value*min_multiplier:
        return sigma_value
    else: 
        return mean_value*min_multiplier


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sem_weighted(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] weights):
    cdef double xW_mean = weighted_mean(values, weights)
    cdef Py_ssize_t N = weights.shape[0]
    cdef double sum_of_weights = sum_weights(weights)
    cdef double w_mean = sum_of_weights/ N
    cdef double[::1] view_v = values
    cdef double[::1] view_w = weights
    cdef double sum1 = 0
    cdef double sum2 = 0
    cdef double sum3 = 0
    for i in range(N):
        sum1 += view_w[i] * view_w[i] * view_v[i] * view_v[i] - 2 * view_w[i] * view_v[i] * w_mean * xW_mean + w_mean * w_mean * xW_mean * xW_mean
        sum2 += view_w[i] * view_w[i] * view_v[i] - view_w[i] * w_mean * xW_mean - view_w[i]* w_mean * view_v[i] + w_mean*w_mean * xW_mean
        sum3 += view_w[i] * view_w[i] - 2* view_w[i] * w_mean + w_mean * w_mean
    return N *  (sum1 -2*xW_mean*sum2 + xW_mean*xW_mean*sum3 ) / ((N-1) *  sum_of_weights* sum_of_weights) 

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sem_weighted_sp(np.ndarray[float, ndim=1] values, np.ndarray[double, ndim=1] weights):
    cdef double xW_mean = weighted_mean_sp(values, weights)
    cdef Py_ssize_t N = weights.shape[0]
    cdef double sum_of_weights = sum_weights(weights)
    cdef double w_mean = sum_of_weights/ N
    cdef float[::1] view_v = values
    cdef double[::1] view_w = weights
    cdef double sum1 = 0
    cdef double sum2 = 0
    cdef double sum3 = 0
    for i in range(N):
        sum1 += view_w[i] * view_w[i] * view_v[i] * view_v[i] - 2 * view_w[i] * view_v[i] * w_mean * xW_mean + w_mean * w_mean * xW_mean * xW_mean
        sum2 += view_w[i] * view_w[i] * view_v[i] - view_w[i] * w_mean * xW_mean - view_w[i]* w_mean * view_v[i] + w_mean*w_mean * xW_mean
        sum3 += view_w[i] * view_w[i] - 2* view_w[i] * w_mean + w_mean * w_mean
    return N *  (sum1 -2*xW_mean*sum2 + xW_mean*xW_mean*sum3 ) / ((N-1) *  sum_of_weights* sum_of_weights) 
