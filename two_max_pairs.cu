#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include "two_max_pairs.cuh"

#define MAX_PAIR_KEY 268435456  // Assuming 14-bit tokens: 16384 * 16384
#define BIT_MASK 0x3FFF  // Mask for 14-bit tokens
#define BIT_OFFSET 14
// #define MAX_PAIR_KEY 1048576  // Assuming 10-bit tokens: 1024 * 1024
// #define MAX_PAIR_KEY 65536  // Assuming 8-bit tokens: 256 * 256

__global__ void count_pair_frequencies_kernel(
    int* data,       // Flattened list of all bites
    const int* offsets,    // Start of each chunk
    const int* lengths,    // Length of each chunk
    const int* num_chunks, // Number of chunks
    int* global_pair_counts // Size: MAX_PAIR_KEY
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
                             int* global_pair_counts, // Size: MAX_PAIR_KEY
                             int* max_pair, // Max pair to replace with new value
                             int* frequency // Frequency of the most frequent pair
) {
    
    int *d_data, *d_offsets, *d_lengths;
    int *d_counts, *cn;

    cudaMalloc(&d_data, num_elements * sizeof(int));
    cudaMalloc(&d_offsets, num_chunks * sizeof(int));
    cudaMalloc(&d_lengths, num_chunks * sizeof(int));
    cudaMalloc(&d_counts, MAX_PAIR_KEY * sizeof(int));
    cudaMalloc(&cn,sizeof(int));

    cudaMemcpy(d_data, data, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, num_chunks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths, num_chunks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cn, &num_chunks, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (num_chunks + threadsPerBlock - 1) / threadsPerBlock;

    count_pair_frequencies_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_offsets, d_lengths, cn, d_counts);
    cudaDeviceSynchronize();

    cudaMemcpy(global_pair_counts, d_counts, MAX_PAIR_KEY * sizeof(int), cudaMemcpyDeviceToHost);

    int max_frequency = 0;

    for (int key = 0; key < MAX_PAIR_KEY; ++key) {
        if (global_pair_counts[key] > 0) {
            // int a = key >> 8;
            // int b = key & 0xFF;
            // int a = key >> 10;  // Adjusted for 10-bit tokens
            // int b = key & 0x3FF;  // Adjusted for 10-bit tokens
            int a = key >> BIT_OFFSET;  // Adjusted for 14-bit tokens
            int b = key & BIT_MASK;  // Adjusted for 14-bit tokens
            // std::cout << "(" << a << ", " << b << ") -> " << global_pair_counts[key] << "\n";
            if (global_pair_counts[key] > max_frequency) {
                max_frequency = global_pair_counts[key];
                max_pair[0] = a;
                max_pair[1] = b;
            }
        }
    }

    frequency[0] = max_frequency;

    cudaFree(d_data);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_counts);
    cudaFree(cn);

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
    // printf("Replacing pair (%d, %d) with new value %d in chunk %d\n", pair[0], pair[1], *new_value);

    for (int i = 0; i < length - 1; ++i) {
        int a = data[offset + i];
        int b = data[offset + i + 1];
        if (a== pair[0] && b == pair[1]) {
            // Replace the pair with the new value
            // printf("Replacing pair (%d, %d) with new value %d in chunk %d at index %d\n", a, b, *new_value, chunk_id, offset + i);
            data[offset + i] = *new_value; // Mark the first element as replaced
            data[offset + i + 1] = -1; // Mark the second element as replaced
            // printf("Replacing pair (%d, %d) with new value %d in chunk %d at index %d\n", a, b, *new_value, chunk_id, offset + i);
            ++i; // Skip the next element since we just replaced it
        }
    }

}

void replace_single_most_frequent(int* data,       // Flattened list of all bites
                                  const int* offsets,    // Start of each chunk
                                  const int* lengths,    // Length of each chunk
                                  const int num_elements, // Number of elements
                                  const int num_chunks, // Number of chunks
                                  int* pair, // Pair to replace
                                  int new_value // New value to replace with
) {
    
    int *d_data, *d_offsets, *d_lengths;
    int *d_counts, *cn, *max_pair, *new_value_cuda;

    cudaMalloc(&d_data, num_elements * sizeof(int));
    cudaMalloc(&d_offsets, num_chunks * sizeof(int));
    cudaMalloc(&d_lengths, num_chunks * sizeof(int));
    cudaMalloc(&d_counts, MAX_PAIR_KEY * sizeof(int));
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

    cudaMemcpy(data, d_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_counts);
    cudaFree(cn);
    cudaFree(max_pair);
    cudaFree(new_value_cuda);

}


void load_txt_input(const std::string& filename,
                    std::vector<int>& flat_data,
                    std::vector<int>& offsets,
                    std::vector<int>& lengths) {
    std::ifstream file(filename);
    std::string line;
    int offset = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int value;
        int count = 0;

        while (iss >> value) {
            flat_data.push_back(value);
            count++;
        }

        offsets.push_back(offset);
        lengths.push_back(count);
        offset += count;
    }
}


int main(int argc, char *argv[]) {
    std::cout << "You have entered " << argc
         << " arguments:" << std::endl;

    std::string filename;

    if (argc > 1) {
        filename = argv[1];
        std::cout << "Using provided filename: " << filename << std::endl;
    } else {
        // const std::string filename = "test_input_redux_mini.txt";
        filename = "test_input_redux_large.txt";
        // const std::string filename = "test_input_redux_smallest.txt";
        std::cout << "No additional arguments provided." << std::endl;
    }

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    std::cout << "Total GPUs: " << gpu_count << std::endl;

    std::vector<int> host_data, host_offsets, host_lengths;
    std::vector<int> host_pairs(MAX_PAIR_KEY);
    std::vector<int> most_frequent_pair = {0, 0};
    int max_frequency = 0;

    load_txt_input(filename, host_data, host_offsets, host_lengths);


    std::cout << "\nLoaded data:\n";

    count_pair_frequencies(host_data.data(), 
                             host_offsets.data(), 
                             host_lengths.data(), 
                             host_data.size(), 
                             host_offsets.size(), 
                             host_pairs.data(),
                             most_frequent_pair.data(),
                             &max_frequency);

    std::cout << "\n";
    // Pring host_pairs
    

    std::cout << "(NEW) Most frequent pair: (" << most_frequent_pair[0] << ", " << most_frequent_pair[1] << ") with frequency " << max_frequency << "\n";
    replace_single_most_frequent(host_data.data(), 
                                  host_offsets.data(), 
                                  host_lengths.data(), 
                                  host_data.size(), 
                                  host_offsets.size(), 
                                  most_frequent_pair.data(), 
                                  256);
    
    // std::cout << "Modified Data:\n";
    // for (size_t i = 0; i < host_offsets.size(); ++i) {
    //     std::cout << "List " << i << ": ";
    //     for (int j = 0; j < host_lengths[i]; ++j) {
    //         int value = host_data[host_offsets[i] + j];
    //         if (value == -1) {
    //             std::cout << "(R) ";
    //         } else {
    //             std::cout << value << " ";
    //         }
    //     }
    //     std::cout << "\n";
    // }


}






int old_main(int argc, char *argv[]) {
    std::cout << "(OLD) You have entered " << argc
         << " arguments:" << std::endl;

    std::string filename;

    if (argc > 1) {
        filename = argv[1];
        std::cout << "Using provided filename: " << filename << std::endl;
    } else {
        // const std::string filename = "test_input_redux_mini.txt";
        filename = "test_input_redux_large.txt";
        // const std::string filename = "test_input_redux_smallest.txt";
        std::cout << "No additional arguments provided." << std::endl;
    }

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    std::cout << "Total GPUs: " << gpu_count << std::endl;

    std::vector<int> host_data, host_offsets, host_lengths;

    load_txt_input(filename, host_data, host_offsets, host_lengths);
    // print host_offsets and host_lengths
    // std::cout << "Offsets: ";
    // for (const auto& offset : host_offsets) {
    //     std::cout << offset << " ";
    // }
    // std::cout << "\nLengths: ";
    // for (const auto& length : host_lengths) {
    //     std::cout << length << " ";
    // }
    std::cout << "\n";
    //print host_data using host_offsets and host_lengths
    // std::cout << "Loaded data:\n";
    // for (size_t i = 0; i < host_offsets.size(); ++i) {
    //     std::cout << "List " << i << ": ";
    //     for (int j = 0; j < host_lengths[i]; ++j) {
    //         std::cout << host_data[host_offsets[i] + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    int num_lists = host_offsets.size();
    int new_value = 256;

    // Allocate and copy inputs to device
    int *d_data, *d_offsets, *d_lengths;
    int *d_counts, *cn, *max_pair, *new_value_cuda;

    cudaMalloc(&d_data, host_data.size() * sizeof(int));
    cudaMalloc(&d_offsets, num_lists * sizeof(int));
    cudaMalloc(&d_lengths, num_lists * sizeof(int));
    cudaMalloc(&d_counts, MAX_PAIR_KEY * sizeof(int));
    cudaMalloc(&cn,sizeof(int));

    cudaMemcpy(d_data, host_data.data(), host_data.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, host_offsets.data(), num_lists * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, host_lengths.data(), num_lists * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cn, &num_lists, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (num_lists + threadsPerBlock - 1) / threadsPerBlock;

    count_pair_frequencies_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_offsets, d_lengths, cn, d_counts);
    cudaDeviceSynchronize();

    // Copy results back
    std::vector<int> host_pairs(MAX_PAIR_KEY);
    cudaMemcpy(host_pairs.data(), d_counts, host_pairs.size() * sizeof(int), cudaMemcpyDeviceToHost);


    std::cout << "Pair Frequencies:\n";
    std::vector<int> most_frequent_pair = {0, 0};
    int max_frequency = 0;
    for (int key = 0; key < MAX_PAIR_KEY; ++key) {
        if (host_pairs[key] > 0) {
            // int a = key >> 8;
            // int b = key & 0xFF;
            // int a = key >> 10;  // Adjusted for 10-bit tokens
            // int b = key & 0x3FF;  // Adjusted for 10-bit tokens
            int a = key >> 14;  // Adjusted for 10-bit tokens
            int b = key & 0x3FFF;  // Adjusted for 10-bit tokens
            std::cout << "(" << a << ", " << b << ") -> " << host_pairs[key] << "\n";
            if (host_pairs[key] > max_frequency) {
                max_frequency = host_pairs[key];
                most_frequent_pair = {a, b};
            }
        }
    }

    cudaMalloc(&max_pair, 2 * sizeof(int));
    cudaMalloc(&new_value_cuda,sizeof(int));

    cudaMemcpy(new_value_cuda, &new_value, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_pair, most_frequent_pair.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Most frequent pair: (" << most_frequent_pair[0] << ", " << most_frequent_pair[1] << ") with frequency " << max_frequency << "\n";

    replace_single_most_frequent_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_offsets, d_lengths, cn, max_pair, new_value_cuda);
    cudaDeviceSynchronize();

    cudaMemcpy(host_data.data(), d_data, host_data.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the modified data
    // std::cout << "Modified Data:\n";
    // for (size_t i = 0; i < host_offsets.size(); ++i) {
    //     std::cout << "List " << i << ": ";
    //     for (int j = 0; j < host_lengths[i]; ++j) {
    //         int value = host_data[host_offsets[i] + j];
    //         if (value == -1) {
    //             std::cout << "(R) ";
    //         } else {
    //             std::cout << value << " ";
    //         }
    //     }
    //     std::cout << "\n";
    // }

    cudaFree(d_data);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_counts);
    cudaFree(cn);
    cudaFree(max_pair);
    cudaFree(new_value_cuda);
    return 0;
}
