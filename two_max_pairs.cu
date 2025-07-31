#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include "two_max_pairs.cuh"

#define MAX_PAIR_KEY 268435456  // Assuming 14-bit tokens: 16384 * 16384
#define BIT_MASK 0x3FFF  // Mask for 14-bit tokens
#define BIT_OFFSET 14
// #define MAX_PAIR_KEY 1048576  // Assuming 10-bit tokens: 1024 * 1024
// #define MAX_PAIR_KEY 65536  // Assuming 8-bit tokens: 256 * 256

typedef int pair_int;

static pair_int* global_pairs_counter = nullptr;
static std::vector<pair_int*> pairs_counters;
static int number_of_gpus = 0;

extern "C" void allocate_pairs_counter() {
    if (!pairs_counters.empty() || global_pairs_counter != nullptr) {
        return;
    }

    int num_gpus;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus == 0) {
        fprintf(stderr, "No CUDA devices found or CUDA error: %s\n", cudaGetErrorString(err));
        number_of_gpus = 0;
        return;
    }
    number_of_gpus = num_gpus;
    printf("Found %d CUDA devices.\n", num_gpus);
    
    pairs_counters.resize(num_gpus);

    err = cudaMalloc(&global_pairs_counter, MAX_PAIR_KEY * sizeof(pair_int));
    if (err != cudaSuccess) {
        cudaSetDevice(0);
        fprintf(stderr, "Failed to allocate global pairs_counter: %s\n", cudaGetErrorString(err));
        global_pairs_counter = nullptr;
        return;
    }
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        err = cudaMalloc(&pairs_counters[i], MAX_PAIR_KEY * sizeof(pair_int));
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate memory for pairs_counter on GPU %d: %s\n", i, cudaGetErrorString(err));
            pairs_counters[i] = nullptr;
            cudaFree(global_pairs_counter);
            global_pairs_counter = nullptr;

            for (int j = 0; j < i; ++j) {
                if (pairs_counters[j] != nullptr) {
                    cudaSetDevice(j);
                    cudaFree(pairs_counters[j]);
                    pairs_counters[j] = nullptr;
                }
            }
        } else {
            printf("Allocated pairs_counter on GPU %d\n", i);
        }
    }
    cudaSetDevice(0); // Reset to the first GPU after allocation
}

extern "C" void* get_pairs_counter() {
    return nullptr;
}

extern "C" void free_pairs_counter() {
    if (!pairs_counters.empty()) {
        for (int i = 0; i < pairs_counters.size(); ++i) {
            if (pairs_counters[i] != nullptr) {
                cudaSetDevice(i);
                cudaFree(pairs_counters[i]);
                pairs_counters[i] = nullptr;
                printf("Freed pairs_counter on GPU %d\n", i);
            }
        }
        pairs_counters.clear();
        number_of_gpus = 0;
        printf("Freed pairs_counter\n");
    }
    cudaSetDevice(0); // Reset to the first GPU after allocation
}


__global__ void count_pair_frequencies_kernel(
    int* data,       // Flattened list of all bites
    const int* offsets,    // Start of each chunk
    const int* lengths,    // Length of each chunk
    const int* num_chunks, // Number of chunks
    pair_int* global_pair_counts // Size: MAX_PAIR_KEY
) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d processing chunk %d (GridDim (%d %d), BlockDim (%d %d))\n", threadIdx.x, chunk_id, gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    if (chunk_id >= *num_chunks) {
        atomicAdd(&global_pair_counts[chunk_id], -1.0*chunk_id);
        return;
    }

    int offset = offsets[chunk_id];
    int length = lengths[chunk_id];
    // printf("Chunk %d is being processed (%d, %d)\n", chunk_id, offset, length);

    // atomicAdd(&global_pair_counts[chunk_id], chunk_id*10000+offsets[chunk_id]*100+lengths[chunk_id]);

    for (int i = 0; i < length - 1; ++i) {
        int a = data[offset + i];
        int b = data[offset + i + 1];
        if (a < 0 || b < 0) {
            break; // Skip negative values
        }
        int key = (a << BIT_OFFSET) | b;  // Flatten (a,b) into single int key

        atomicAdd(&global_pair_counts[key], 1);
    }
}

void count_pair_frequencies(int* data,       // Flattened list of all bites
                             const int* offsets,    // Start of each chunk
                             const int* lengths,    // Length of each chunk
                             const int num_elements, // Number of elements
                             const int num_chunks, // Number of chunks
                             int* max_pair, // Max pair to replace with new value
                             int* frequency // Frequency of the most frequent pair
) {

    if (number_of_gpus == 0) {
        fprintf(stderr, "No CUDA devices found. Cannot count pair frequencies.\n");
        return;
    }
    
    std::vector<cudaStream_t> streams(number_of_gpus);
    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    std::vector<int*> d_data_vec(number_of_gpus), d_offsets_vec(number_of_gpus), d_lengths_vec(number_of_gpus), d_num_chunks_vec(number_of_gpus);
    std::vector<int> h_num_chunks_vec(number_of_gpus);

    int chunk_size = (num_chunks + number_of_gpus - 1) / number_of_gpus; // Divide chunks evenly across GPUs

    for (int i = 0; i < number_of_gpus; ++i) {
        int start_chunk = i * chunk_size;
        int end_chunk = std::min(start_chunk + chunk_size, num_chunks);
        int num_chunks_for_gpu = end_chunk - start_chunk;

        if (num_chunks_for_gpu <= 0) continue;

        h_num_chunks_vec[i] = num_chunks_for_gpu;

        int start_offset = offsets[start_chunk];
        int end_offset = (end_chunk > 0) ? (offsets[end_chunk - 1] + lengths[end_chunk - 1]) : start_offset;
        int num_elements_for_gpu = end_offset - start_offset;

        if (num_chunks_for_gpu == 0 || num_elements_for_gpu == 0) {
            h_num_chunks_vec[i] = 0;
            continue;
        }

        cudaSetDevice(i);
        cudaMalloc(&d_data_vec[i], num_elements_for_gpu * sizeof(int));
        cudaMalloc(&d_offsets_vec[i], num_chunks_for_gpu * sizeof(int));
        cudaMalloc(&d_lengths_vec[i], num_chunks_for_gpu * sizeof(int));
        cudaMalloc(&d_num_chunks_vec[i], sizeof(int));

        cudaMemcpyAsync(d_data_vec[i], data + start_offset, num_elements_for_gpu * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_offsets_vec[i], offsets + start_chunk, num_chunks_for_gpu * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_lengths_vec[i], lengths + start_chunk, num_chunks_for_gpu * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_num_chunks_vec[i], &h_num_chunks_vec[i], sizeof(int), cudaMemcpyHostToDevice, streams[i]);

        cudaMemsetAsync(pairs_counters[i], 0, MAX_PAIR_KEY * sizeof(pair_int), streams[i]);

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_chunks_for_gpu + threadsPerBlock - 1) / threadsPerBlock;
        // std::cout << "Launching kernel with\n" ;

        count_pair_frequencies_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_data_vec[i], d_offsets_vec[i], d_lengths_vec[i], d_num_chunks_vec[i], pairs_counters[i]);
    }

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error on GPU %d: %s\n", i, cudaGetErrorString(err));
        }
    }

    std::vector<pair_int> h_global_counter(MAX_PAIR_KEY, 0);
    std::vector<pair_int> h_temp_counter(MAX_PAIR_KEY);


    for (int i=0; i < number_of_gpus; ++i) {
        if (h_num_chunks_vec[i] <= 0) continue;

        cudaSetDevice(i);
        cudaMemcpy(h_temp_counter.data(), pairs_counters[i], MAX_PAIR_KEY * sizeof(pair_int), cudaMemcpyDeviceToHost);

        for (int j = 0; j < MAX_PAIR_KEY; ++j) {
            h_global_counter[j] += h_temp_counter[j];
        }
    }

    // Find the maximum element in the global counter
    auto max_it = std::max_element(h_global_counter.begin(), h_global_counter.end());
    if (max_it != h_global_counter.end()) {
        int max_index = std::distance(h_global_counter.begin(), max_it);
        int max_value = *max_it;
        max_pair[0] = max_index >> BIT_OFFSET;  // Extract first token
        max_pair[1] = max_index & BIT_MASK;     // Extract second token
        *frequency = max_value;
    } else {
        max_pair[0] = 0;
        max_pair[1] = 0;
        *frequency = -1;
    }

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        if (h_num_chunks_vec[i] > 0) {
            cudaFree(d_data_vec[i]);
            cudaFree(d_offsets_vec[i]);
            cudaFree(d_lengths_vec[i]);
            cudaFree(d_num_chunks_vec[i]);
        }
        cudaStreamDestroy(streams[i]);
    }

}

__global__ void compact_kernel(int* data, const int* offsets, int* lengths, const int* num_chunks) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= *num_chunks) {
        return;
    }

    int offset = offsets[chunk_id];
    int length = lengths[chunk_id];
    if (length == 0) return;

    int write_ptr = offset;
    for (int read_ptr = offset; read_ptr < offset + length; ++read_ptr) {
        if (data[read_ptr] != -1) {
            if (write_ptr != read_ptr) {
                data[write_ptr] = data[read_ptr];
            }
            write_ptr++;
        }
    }
    int new_length = write_ptr - offset;

    // Pad rest of original chunk with -1
    for (int i = write_ptr; i < offset + length; ++i) {
        data[i] = -1;
    }
    lengths[chunk_id] = new_length;
}

__global__ void replace_single_most_frequent_kernel(
    int* data,       // Flattened list of all bites
    const int* offsets,    // Start of each chunk
    const int* lengths,    // Length of each chunk
    const int* num_chunks, // Number of chunks
    int* pair, // Pair to replace
    int* new_value // New value to replace with
) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("Thread %d processing chunk %d (GridDim (%d %d), BlockDim (%d %d)) %d\n", threadIdx.x, chunk_id, gridDim.x, gridDim.y, blockDim.x, blockDim.y, *num_chunks);

    if (chunk_id >= *num_chunks) {
        return;
    }

    int offset = offsets[chunk_id];
    int length = lengths[chunk_id];
    // printf("Chunk %d is being processed (%d, %d)\n", chunk_id, offset, length);


    for (int i = 0; i < length - 1; ++i) {
        int a = data[offset + i];
        int b = data[offset + i + 1];
        if (a == pair[0] && b == pair[1]) {
            // Replace the pair with the new value
            // printf("Replacing pair (%d, %d) with new value %d in chunk %d at index %d\n", a, b, *new_value, chunk_id, offset + i);
            data[offset + i] = *new_value; // Mark the first element as replaced
            data[offset + i + 1] = -1; // Mark the second element as replaced
            // printf("Replacing pair (%d, %d) with new value %d in chunk %d at index %d\n", a, b, *new_value, chunk_id, offset + i);
            ++i; // Skip the next element since we just replaced it
        }
    }

}

void old_replace_single_most_frequent(int* data,       // Flattened list of all bites
                                  const int* offsets,    // Start of each chunk
                                  const int* lengths,    // Length of each chunk
                                  const int num_elements, // Number of elements
                                  const int num_chunks, // Number of chunks
                                  int* pair, // Pair to replace
                                  int new_value // New value to replace with
) {
    
    int *d_data, *d_offsets, *d_lengths;
    int *cn, *max_pair, *new_value_cuda;

    cudaMalloc(&d_data, num_elements * sizeof(int));
    cudaMalloc(&d_offsets, num_chunks * sizeof(int));
    cudaMalloc(&d_lengths, num_chunks * sizeof(int));
    cudaMalloc(&max_pair, 2 * sizeof(int));
    cudaMalloc(&cn,sizeof(int));
    cudaMalloc(&new_value_cuda,sizeof(int));
    
    cudaMemcpy(d_data, data, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, num_chunks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths, num_chunks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cn, &num_chunks, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_pair, pair, 2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(new_value_cuda, &new_value, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (num_chunks + threadsPerBlock - 1) / threadsPerBlock;

    replace_single_most_frequent_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_offsets, d_lengths, cn, max_pair, new_value_cuda);
    cudaDeviceSynchronize();

    compact_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_offsets, d_lengths, cn);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(cn);
    cudaFree(max_pair);
    cudaFree(new_value_cuda);

}

void replace_single_most_frequent(int* data,       // Flattened list of all bites
                                  const int* offsets,    // Start of each chunk
                                  const int* lengths,    // Length of each chunk
                                  const int num_elements, // Number of elements
                                  const int num_chunks, // Number of chunks
                                  int* pair, // Pair to replace
                                  int new_value // New value to replace with
) {
    
    if (number_of_gpus == 0) {
        fprintf(stderr, "No CUDA devices found. Cannot count pair frequencies.\n");
        return;
    }
    
    std::vector<cudaStream_t> streams(number_of_gpus);
    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    std::vector<int*> d_data_vec(number_of_gpus), d_offsets_vec(number_of_gpus), d_lengths_vec(number_of_gpus), d_num_chunks_vec(number_of_gpus);
    std::vector<int*> d_pair_vec(number_of_gpus), d_new_value_vec(number_of_gpus);
    std::vector<int> h_num_chunks_vec(number_of_gpus);

    int chunk_size = (num_chunks + number_of_gpus - 1) / number_of_gpus; // Divide chunks evenly across GPUs

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);

        int start_chunk = i * chunk_size;
        int end_chunk = std::min(start_chunk + chunk_size, num_chunks);
        int num_chunks_for_gpu = end_chunk - start_chunk;

        if (num_chunks_for_gpu <= 0) {
            h_num_chunks_vec[i] = 0;
            continue;
        }

        h_num_chunks_vec[i] = num_chunks_for_gpu;

        int start_offset = offsets[start_chunk];
        int end_offset = (end_chunk > 0) ? (offsets[end_chunk - 1] + lengths[end_chunk - 1]) : start_offset;
        int num_elements_for_gpu = end_offset - start_offset;

        if (num_chunks_for_gpu <= 0 || num_elements_for_gpu <= 0) {
            h_num_chunks_vec[i] = 0;
            continue;
        }

        cudaMalloc(&d_data_vec[i], num_elements_for_gpu * sizeof(int));
        cudaMalloc(&d_offsets_vec[i], num_chunks_for_gpu * sizeof(int));
        cudaMalloc(&d_lengths_vec[i], num_chunks_for_gpu * sizeof(int));
        cudaMalloc(&d_num_chunks_vec[i], sizeof(int));
        cudaMalloc(&d_pair_vec[i], 2 * sizeof(int));
        cudaMalloc(&d_new_value_vec[i], sizeof(int));

        cudaMemcpyAsync(d_data_vec[i], data + start_offset, num_elements_for_gpu * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_offsets_vec[i], offsets + start_chunk, num_chunks_for_gpu * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_lengths_vec[i], lengths + start_chunk, num_chunks_for_gpu * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_num_chunks_vec[i], &h_num_chunks_vec[i], sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_pair_vec[i], pair, 2 * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_new_value_vec[i], &new_value, sizeof(int), cudaMemcpyHostToDevice, streams[i]);

        int threadsPerBlock = 128;
        int blocksPerGrid = (num_chunks_for_gpu + threadsPerBlock - 1) / threadsPerBlock;

        replace_single_most_frequent_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_data_vec[i], d_offsets_vec[i], d_lengths_vec[i], d_num_chunks_vec[i], d_pair_vec[i], d_new_value_vec[i]);
        compact_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_data_vec[i], d_offsets_vec[i], d_lengths_vec[i], d_num_chunks_vec[i]);
    }

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error on GPU %d: %s\n", i, cudaGetErrorString(err));
        }

        int start_chunk = i * chunk_size;
        int num_chunks_for_gpu = h_num_chunks_vec[i];
        if (num_chunks_for_gpu <= 0) {
            continue;
        }

        int start_offset = offsets[start_chunk];
        int end_offset = (start_chunk + num_chunks_for_gpu < num_chunks) ? (offsets[start_chunk + num_chunks_for_gpu - 1] + lengths[start_chunk + num_chunks_for_gpu - 1]) : start_offset;
        int num_elements_for_gpu = end_offset - start_offset;
        printf("Copying data from GPU %d, start_offset: %d, num_elements_for_gpu: %d\n", i, start_offset, num_elements_for_gpu);
        cudaMemcpy(data + start_offset, d_data_vec[i], num_elements_for_gpu * sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        if (h_num_chunks_vec[i] > 0) {
            cudaFree(d_data_vec[i]);
            cudaFree(d_offsets_vec[i]);
            cudaFree(d_lengths_vec[i]);
            cudaFree(d_num_chunks_vec[i]);
            cudaFree(d_pair_vec[i]);
            cudaFree(d_new_value_vec[i]);
        }
        cudaStreamDestroy(streams[i]);
    }

}


// void load_txt_input(const std::string& filename,
//                     std::vector<int>& flat_data,
//                     std::vector<int>& offsets,
//                     std::vector<int>& lengths) {
//     std::ifstream file(filename);
//     std::string line;
//     int offset = 0;

//     while (std::getline(file, line)) {
//         std::istringstream iss(line);
//         int value;
//         int count = 0;

//         while (iss >> value) {
//             flat_data.push_back(value);
//             count++;
//         }

//         offsets.push_back(offset);
//         lengths.push_back(count);
//         offset += count;
//     }
// }


// int main(int argc, char *argv[]) {
//     std::cout << "You have entered " << argc
//          << " arguments:" << std::endl;

//     std::string filename;

//     if (argc > 1) {
//         filename = argv[1];
//         std::cout << "Using provided filename: " << filename << std::endl;
//     } else {
//         // const std::string filename = "test_input_redux_mini.txt";
//         filename = "test_input_redux_large.txt";
//         // const std::string filename = "test_input_redux_smallest.txt";
//         std::cout << "No additional arguments provided." << std::endl;
//     }

//     int gpu_count;
//     cudaGetDeviceCount(&gpu_count);
//     std::cout << "Total GPUs: " << gpu_count << std::endl;

//     std::vector<int> host_data, host_offsets, host_lengths;
//     std::vector<int> host_pairs(MAX_PAIR_KEY);
//     std::vector<int> most_frequent_pair = {0, 0};
//     int max_frequency = 0;

//     load_txt_input(filename, host_data, host_offsets, host_lengths);


//     std::cout << "\nLoaded data:\n";

//     count_pair_frequencies(host_data.data(), 
//                              host_offsets.data(), 
//                              host_lengths.data(), 
//                              host_data.size(), 
//                              host_offsets.size(), 
//                              most_frequent_pair.data(),
//                              &max_frequency);

//     std::cout << "\n";
//     // Pring host_pairs
    

//     std::cout << "(NEW) Most frequent pair: (" << most_frequent_pair[0] << ", " << most_frequent_pair[1] << ") with frequency " << max_frequency << "\n";
//     replace_single_most_frequent(host_data.data(), 
//                                   host_offsets.data(), 
//                                   host_lengths.data(), 
//                                   host_data.size(), 
//                                   host_offsets.size(), 
//                                   most_frequent_pair.data(), 
//                                   256);
    
//     // std::cout << "Modified Data:\n";
//     // for (size_t i = 0; i < host_offsets.size(); ++i) {
//     //     std::cout << "List " << i << ": ";
//     //     for (int j = 0; j < host_lengths[i]; ++j) {
//     //         int value = host_data[host_offsets[i] + j];
//     //         if (value == -1) {
//     //             std::cout << "(R) ";
//     //         } else {
//     //             std::cout << value << " ";
//     //         }
//     //     }
//     //     std::cout << "\n";
//     // }


// }






// int old_main(int argc, char *argv[]) {
//     std::cout << "(OLD) You have entered " << argc
//          << " arguments:" << std::endl;

//     std::string filename;

//     if (argc > 1) {
//         filename = argv[1];
//         std::cout << "Using provided filename: " << filename << std::endl;
//     } else {
//         // const std::string filename = "test_input_redux_mini.txt";
//         filename = "test_input_redux_large.txt";
//         // const std::string filename = "test_input_redux_smallest.txt";
//         std::cout << "No additional arguments provided." << std::endl;
//     }

//     int gpu_count;
//     cudaGetDeviceCount(&gpu_count);
//     std::cout << "Total GPUs: " << gpu_count << std::endl;

//     std::vector<int> host_data, host_offsets, host_lengths;

//     load_txt_input(filename, host_data, host_offsets, host_lengths);
//     // print host_offsets and host_lengths
//     // std::cout << "Offsets: ";
//     // for (const auto& offset : host_offsets) {
//     //     std::cout << offset << " ";
//     // }
//     // std::cout << "\nLengths: ";
//     // for (const auto& length : host_lengths) {
//     //     std::cout << length << " ";
//     // }
//     std::cout << "\n";
//     //print host_data using host_offsets and host_lengths
//     // std::cout << "Loaded data:\n";
//     // for (size_t i = 0; i < host_offsets.size(); ++i) {
//     //     std::cout << "List " << i << ": ";
//     //     for (int j = 0; j < host_lengths[i]; ++j) {
//     //         std::cout << host_data[host_offsets[i] + j] << " ";
//     //     }
//     //     std::cout << "\n";
//     // }

//     int num_lists = host_offsets.size();
//     int new_value = 256;

//     // Allocate and copy inputs to device
//     int *d_data, *d_offsets, *d_lengths;
//     int *d_counts, *cn, *max_pair, *new_value_cuda;

//     cudaMalloc(&d_data, host_data.size() * sizeof(int));
//     cudaMalloc(&d_offsets, num_lists * sizeof(int));
//     cudaMalloc(&d_lengths, num_lists * sizeof(int));
//     cudaMalloc(&d_counts, MAX_PAIR_KEY * sizeof(int));
//     cudaMalloc(&cn,sizeof(int));

//     cudaMemcpy(d_data, host_data.data(), host_data.size() * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_offsets, host_offsets.data(), num_lists * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_lengths, host_lengths.data(), num_lists * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(cn, &num_lists, sizeof(int), cudaMemcpyHostToDevice);

//     // Launch kernel
//     int threadsPerBlock = 128;
//     int blocksPerGrid = (num_lists + threadsPerBlock - 1) / threadsPerBlock;

//     count_pair_frequencies_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_offsets, d_lengths, cn, d_counts);
//     cudaDeviceSynchronize();

//     // Copy results back
//     std::vector<int> host_pairs(MAX_PAIR_KEY);
//     cudaMemcpy(host_pairs.data(), d_counts, host_pairs.size() * sizeof(int), cudaMemcpyDeviceToHost);


//     std::cout << "Pair Frequencies:\n";
//     std::vector<int> most_frequent_pair = {0, 0};
//     int max_frequency = 0;
//     for (int key = 0; key < MAX_PAIR_KEY; ++key) {
//         if (host_pairs[key] > 0) {
//             // int a = key >> 8;
//             // int b = key & 0xFF;
//             // int a = key >> 10;  // Adjusted for 10-bit tokens
//             // int b = key & 0x3FF;  // Adjusted for 10-bit tokens
//             int a = key >> 14;  // Adjusted for 10-bit tokens
//             int b = key & 0x3FFF;  // Adjusted for 10-bit tokens
//             std::cout << "(" << a << ", " << b << ") -> " << host_pairs[key] << "\n";
//             if (host_pairs[key] > max_frequency) {
//                 max_frequency = host_pairs[key];
//                 most_frequent_pair = {a, b};
//             }
//         }
//     }

//     cudaMalloc(&max_pair, 2 * sizeof(int));
//     cudaMalloc(&new_value_cuda,sizeof(int));

//     cudaMemcpy(new_value_cuda, &new_value, sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(max_pair, most_frequent_pair.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

//     // std::cout << "Most frequent pair: (" << most_frequent_pair[0] << ", " << most_frequent_pair[1] << ") with frequency " << max_frequency << "\n";

//     replace_single_most_frequent_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_offsets, d_lengths, cn, max_pair, new_value_cuda);
//     cudaDeviceSynchronize();

//     cudaMemcpy(host_data.data(), d_data, host_data.size() * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print the modified data
//     // std::cout << "Modified Data:\n";
//     // for (size_t i = 0; i < host_offsets.size(); ++i) {
//     //     std::cout << "List " << i << ": ";
//     //     for (int j = 0; j < host_lengths[i]; ++j) {
//     //         int value = host_data[host_offsets[i] + j];
//     //         if (value == -1) {
//     //             std::cout << "(R) ";
//     //         } else {
//     //             std::cout << value << " ";
//     //         }
//     //     }
//     //     std::cout << "\n";
//     // }

//     cudaFree(d_data);
//     cudaFree(d_offsets);
//     cudaFree(d_lengths);
//     cudaFree(d_counts);
//     cudaFree(cn);
//     cudaFree(max_pair);
//     cudaFree(new_value_cuda);
//     return 0;
// }
