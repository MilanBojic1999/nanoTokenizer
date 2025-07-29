from libc.stdlib cimport malloc, free
cdef extern from "two_max_pairs.cuh":
    void count_pair_frequencies(int* data, const int* offsets, const int* lengths, const int num_elements, const int num_chunks, int* global_pair_counts, int* max_pair, int* frequency )

    void replace_single_most_frequent(int* data, const int* offsets, const int* lengths, const int num_elements, const int num_chunks, int* pair, int new_value )

    int MAX_PAIR_KEY
    int BIT_MASK
    int BIT_OFFSET

import numpy as np
cimport numpy as np

def cuda_count_pair_frequencies(np.ndarray[np.int32_t, ndim=2] data):

    cdef np.ndarray[np.int32_t, ndim=1] flat_data = data.flatten()
    cdef int num_elements = flat_data.shape[0]
    cdef int num_chunks = data.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] offsets = np.zeros(num_chunks, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] lengths = np.zeros(num_chunks, dtype=np.int32)

    cdef int current_offset = 0
    for ind,el in enumerate(data):
        offsets[ind] = current_offset
        lengths[ind] = el.shape[0]
        current_offset += el.shape[0]

    cdef np.ndarray[np.int32_t, ndim=1] global_pair_counts = np.zeros(MAX_PAIR_KEY, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] max_pair = np.zeros(2, dtype=np.int32)
    cdef int *frequency = <int*>malloc(sizeof(int))

    count_pair_frequencies(<int*>flat_data.data, <int*>offsets.data, <int*>lengths.data, num_elements, num_chunks, <int*>global_pair_counts.data, <int*>max_pair.data, frequency)
    
    # print(type(frequency))
    # print(global_pair_counts)
    cdef np.int32_t ret_frequency = frequency[0]
    free(frequency)

    return max_pair, ret_frequency


def move_neg_ones_to_end(arr):
    result = []
    for row in arr:
        row = np.array(row)
        non_neg_ones = row[row != -1]
        neg_ones = row[row == -1]
        new_row = np.concatenate([non_neg_ones, neg_ones])
        result.append(new_row)
    return np.array(result)


def cuda_replace_single_most_frequent(np.ndarray[np.int32_t, ndim=2] data, np.ndarray[np.int32_t, ndim=1] pair, int new_value):

    cdef np.ndarray[np.int32_t, ndim=1] flat_data = data.flatten()
    cdef int num_elements = flat_data.shape[0]
    cdef int num_chunks = data.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] offsets = np.zeros(num_chunks, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] lengths = np.zeros(num_chunks, dtype=np.int32)

    cdef int current_offset = 0
    for ind,el in enumerate(data):
        offsets[ind] = current_offset
        lengths[ind] = el.shape[0]
        current_offset += el.shape[0]

    replace_single_most_frequent(<int*>flat_data.data, <int*>offsets.data, <int*>lengths.data, num_elements, num_chunks, <int*>pair.data, new_value)

    cdef np.ndarray[np.int32_t, ndim=2] new_data

    cdef tuple shape_tuple = tuple([data.shape[0], data.shape[1]])


    tmp_data = flat_data.reshape(shape_tuple)
    new_data = tmp_data

    # new_data = np.delete(new_data, np.where(new_data == -1)[0])
    new_data = move_neg_ones_to_end(new_data)
    return new_data

    


    



