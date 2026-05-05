from libc.stdlib cimport malloc, free
cdef extern from "two_max_pairs.cuh":

    void allocate_elemets(int* data, const int* offsets, const int* lengths, const int num_elements, const int num_chunks)
    void free_elemets()
    void get_data(int* data, int* num_elements)

    void count_pair_frequencies(int* max_pair, int* frequency)
    void replace_single_most_frequent(int* pair, int new_value)

    int MAX_PAIR_KEY
    int BIT_MASK
    int BIT_OFFSET

import numpy as np
cimport numpy as np

def allocate_cuda_elemets(
    np.ndarray[np.int32_t, ndim=1] data,
    np.ndarray[np.int32_t, ndim=1] offsets,
    np.ndarray[np.int32_t, ndim=1] lengths
):
    # Use typed copies so .data can be cast to int*
    cdef np.ndarray[np.int32_t, ndim=1] c_data    = np.copy(data)
    cdef np.ndarray[np.int32_t, ndim=1] c_offsets = np.copy(offsets)
    cdef np.ndarray[np.int32_t, ndim=1] c_lengths = np.copy(lengths)
    cdef int num_elements = data.shape[0]    # was mistakenly passed as c_lengths
    cdef int num_chunks   = offsets.shape[0]

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
    cdef np.ndarray[np.int32_t, ndim=1] c_data    = np.copy(data)
    cdef np.ndarray[np.int32_t, ndim=1] c_lengths = np.copy(lengths)

    get_data(<int*>c_data.data, <int*>c_lengths.data)
    data[:] = c_data[:c_lengths[0]]


def cuda_count_pair_frequencies(
    np.ndarray[np.int32_t, ndim=1] pair,
    np.ndarray[np.int32_t, ndim=1] pair_frequencies
):
    count_pair_frequencies(<int*>pair.data, <int*>pair_frequencies.data)


def cuda_replace_single_most_frequent(
    np.ndarray[np.int32_t, ndim=1] pair,
    int new_value
):
    # new_value is a plain int — pass its address, not a cast
    replace_single_most_frequent(<int*>pair.data, new_value)