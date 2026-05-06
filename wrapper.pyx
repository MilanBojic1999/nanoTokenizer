from libc.stdlib cimport malloc, free
cdef extern from "two_max_pairs.cuh":

    void allocate_elemets(int* data, const int* offsets, const int* lengths, const int num_elements, const int num_chunks)
    void free_elemets()
    void get_data(int* data, int* num_elements)

    void count_pair_frequencies(int* max_pair, int* frequency)
    void replace_single_most_frequent(int* pair, int new_value)


import numpy as np
cimport numpy as np

def allocate_cuda_elemets(
    np.ndarray[np.int32_t, ndim=1] data,
    np.ndarray[np.int32_t, ndim=1] offsets,
    np.ndarray[np.int32_t, ndim=1] lengths
):
    # Use typed copies so .data can be cast to int*
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] c_data    = np.ascontiguousarray(data)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] c_offsets = np.ascontiguousarray(offsets)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] c_lengths = np.ascontiguousarray(lengths)
    cdef int num_elements = data.shape[0]    # was mistakenly passed as c_lengths
    cdef int num_chunks   = lengths.shape[0]

    allocate_elemets(
        <int*>c_data.data,
        <int*>c_offsets.data,
        <int*>c_lengths.data,
        num_elements,   # fixed: was c_lengths (the array)
        num_chunks
    )


def free_cuda_elemets():
    free_elemets()


def get_current_data(
    np.ndarray[np.int32_t, ndim=1] data,
    np.ndarray[np.int32_t, ndim=1] lengths
):
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] c_data    = np.ascontiguousarray(data)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] c_lengths = np.ascontiguousarray(lengths)
    get_data(<int*>c_data.data, <int*>c_lengths.data)

    data[:] = c_data
    lengths[:] = c_lengths



def cuda_count_pair_frequencies(
    np.ndarray[np.int32_t, ndim=1] pair,
    np.ndarray[np.int32_t, ndim=1] pair_frequencies
):
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] c_pair = np.ascontiguousarray(pair)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] c_pair_frequencies = np.ascontiguousarray(pair_frequencies)
    count_pair_frequencies(<int*>c_pair.data, <int*>c_pair_frequencies.data)


def cuda_replace_single_most_frequent(
    np.ndarray[np.int32_t, ndim=1] pair,
    int new_value
):
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] c_pair = np.ascontiguousarray(pair)
    # new_value is a plain int — pass its address, not a cast
    replace_single_most_frequent(<int*>c_pair.data, new_value)