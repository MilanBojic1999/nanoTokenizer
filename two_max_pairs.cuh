#ifndef TWO_MAX_PAIRS_CUH
#define TWO_MAX_PAIRS_CUH

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define MAX_PAIR_KEY 268435456  // Assuming 14-bit tokens: 16384 * 16384
#define BIT_MASK 0x3FFF  // Mask for 14-bit tokens
#define BIT_OFFSET 14

extern "C" void allocate_pairs_counter();
extern "C" void* get_pairs_counter();
extern "C" void free_pairs_counter();


void count_pair_frequencies(int* data,       // Flattened list of all bites
                             const int* offsets,    // Start of each chunk
                             const int* lengths,    // Length of each chunk
                             const int num_elements, // Number of elements
                             const int num_chunks, // Number of chunks
                             int* max_pair, // Max pair to replace with new value
                             int* frequency // Frequency of the max pair

);

void replace_single_most_frequent(int* data,       // Flattened list of all bites
                                  const int* offsets,    // Start of each chunk
                                  const int* lengths,    // Length of each chunk
                                  const int num_elements, // Number of elements
                                  const int num_chunks, // Number of chunks
                                  int* pair, // Pair to replace
                                  int new_value // New value to replace with
);

#endif