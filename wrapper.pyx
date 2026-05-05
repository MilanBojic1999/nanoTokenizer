from libc.stdlib cimport malloc, free
cdef extern from "two_max_pairs.cuh":

    void allocate_cuda_elemets(int* data, const int* offsets, const int* lengths, const int num_elements, const int num_chunks)
    void free_cuda_elemets()
    void get_data(int* data, int* num_elements)

    void count_pair_frequencies(int* max_pair, int* frequency)

    void replace_single_most_frequent(int* max_pair, int* frequency)

    int MAX_PAIR_KEY
    int BIT_MASK
    int BIT_OFFSET

import numpy as np
cimport numpy as np

def allocate_cuda_elemets(np.ndarray[np.int32_t, ndim=1] data, np.ndarray[np.int32_t, ndim=1] offsets, np.ndarray[np.int32_t, ndim=1] lengths):
    """
    Allocate global arrays for pair counting.
    This function should be called before using any CUDA functions that require these arrays.
    """

    allocate_cuda_elemets(
        <int*>data.data, 
        <int*>offsets.data, 
        <int*>lengths.data, 
        data.shape[0], 
        offsets.shape[0]
    )


def free_cuda_elemets():
    """Free the global arrays allocated for pair counting.
    This function should be called when the arrays are no longer needed to avoid memory leaks.
    """
    free_cuda_elemets()

def get_current_data(np.ndarray[np.int32_t, ndim=1] data, np.ndarray[np.int32_t, ndim=1] lengths):
    """Get the current data from the CUDA global arrays.
    This function can be used for debugging or to inspect the current state of the data on the GPU.
    Returns:
        A tuple containing a numpy array: data.
    """
    cdef int num_elements = data.shape[0]
    get_data(<int*>data.data, <int*>lengths.data)

def cuda_count_pair_frequencies(np.ndarray[np.int32_t, ndim=1] pair, int np.ndarray[np.int32_t, ndim=1] pair_frequencies):

    cdef np.ndarray[np.int32_t, ndim=1] max_pair = np.zeros(2, dtype=np.int32)
    cdef int *frequency = <int*>malloc(sizeof(int))

    count_pair_frequencies(<int*>pair.data, <int*>pair_frequencies.data)


def cuda_replace_single_most_frequent(np.ndarray[np.int32_t, ndim=1] pair, int new_value):


    replace_single_most_frequent(<int*>pair.data, new_value)
