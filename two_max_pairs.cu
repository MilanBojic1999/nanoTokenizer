#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <fstream>
#include <sstream>
#include "two_max_pairs.cuh"


#define MAX_PAIR_KEY 268435456  // Assuming 14-bit tokens: 16384 * 16384
#define BIT_MASK 0x3FFF  // Mask for 14-bit tokens
#define BIT_OFFSET 14
// #define MAX_PAIR_KEY 1048576  // Assuming 10-bit tokens: 1024 * 1024
// #define MAX_PAIR_KEY 65536  // Assuming 8-bit tokens: 256 * 256

typedef int pair_int;

typedef struct
{
    int* d_data;
    int* d_offsets;
    int* d_lengths;
    pair_int* d_pairs_counter;

    int start_chunk;
    int end_chunk;
    int element_start;
    int num_elements;

} GPUState;



static int number_of_gpus = 0;
static int num_elements = 0;
static int num_chunks = 0;
static GPUState* gpu_states;

extern "C" void allocate_elemets(int* data,       // Flattened list of all bites
                             const int* offsets,    // Start of each chunk
                             const int* lengths,    // Length of each chunk
                             const int host_num_elements, // Number of elements
                             const int host_num_chunks // Number of chunks
) {

    if (gpu_states != nullptr) {
        fprintf(stderr, "GPU states already allocated. Freeing existing GPU states before reallocation.\n");
        free_elemets();
    }

    int num_gpus;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus == 0) {
        fprintf(stderr, "No CUDA devices found or CUDA error: %s\n", cudaGetErrorString(err));
        number_of_gpus = 0;
        return;
    }
    number_of_gpus = num_gpus;
    num_elements = host_num_elements;
    num_chunks = host_num_chunks;
    printf("Found %d CUDA devices.\n", num_gpus);
    

    gpu_states = (GPUState*)malloc(num_gpus * sizeof(GPUState));
    int tokens_per_gpu = (num_elements + num_gpus - 1) / num_gpus;
    int token_accumulated = 0;
    int current_gpu = 0;

    gpu_states[0].start_chunk = 0;
    gpu_states[0].element_start = 0;

    // Assign chunks to GPUs based on the number of tokens, ensuring that each GPU gets a roughly equal number of tokens to process
    for (int i = 0; i < num_chunks && current_gpu < number_of_gpus - 1; ++i) {
        token_accumulated += lengths[i];
        if (token_accumulated >= (current_gpu + 1) * tokens_per_gpu) {
            gpu_states[current_gpu].end_chunk = i + 1;
            ++current_gpu;
            gpu_states[current_gpu].start_chunk = i + 1;
            gpu_states[current_gpu].element_start = offsets[i + 1];
        }
    }

    gpu_states[number_of_gpus - 1].end_chunk = num_chunks;
    if (number_of_gpus > 1) {
        int last_start_chunk = gpu_states[number_of_gpus - 1].start_chunk;
        gpu_states[number_of_gpus - 1].element_start = offsets[last_start_chunk];
    }

    for (int i = 0; i < num_gpus; ++i) {
        int gpu_start_chunk  = gpu_states[i].start_chunk;
        int gpu_end_chunk    = gpu_states[i].end_chunk;
        int gpu_num_chunks   = gpu_end_chunk - gpu_start_chunk;
        int gpu_elem_start   = gpu_states[i].element_start;
        int total_offset_end = (gpu_end_chunk == num_chunks) ? host_num_elements : offsets[gpu_end_chunk];
        int gpu_num_elements = total_offset_end - offsets[gpu_start_chunk];
        
        gpu_states[i].num_elements = gpu_num_elements;
        printf("GPU chunk assignments %d): %d || %d || %d || %d >> %d\n", i, gpu_start_chunk, gpu_end_chunk, gpu_num_chunks, gpu_elem_start, gpu_num_elements);
        
        if (gpu_num_chunks <= 0 || gpu_num_elements <= 0) {
            gpu_states[i].d_data = nullptr;
            gpu_states[i].d_offsets = nullptr;
            gpu_states[i].d_lengths = nullptr;
            gpu_states[i].d_pairs_counter = nullptr;
            continue;
        }

        cudaSetDevice(i);
        cudaMalloc(&gpu_states[i].d_data, gpu_num_elements * sizeof(int));
        cudaMalloc(&gpu_states[i].d_offsets, gpu_num_chunks * sizeof(int));
        cudaMalloc(&gpu_states[i].d_lengths, gpu_num_chunks * sizeof(int));
        cudaMalloc(&gpu_states[i].d_pairs_counter, MAX_PAIR_KEY * sizeof(pair_int));

        cudaMemcpy(gpu_states[i].d_data, data + gpu_elem_start, gpu_num_elements * sizeof(int), cudaMemcpyHostToDevice);

        // Moving offsets to be relative to the GPU's assigned chunks
        int* relative_offsets = (int*)malloc(gpu_num_chunks * sizeof(int));
        for (int j = 0; j < gpu_num_chunks; ++j) {
            relative_offsets[j] = offsets[gpu_start_chunk + j] - gpu_elem_start;
        }
        cudaMemcpy(gpu_states[i].d_offsets, relative_offsets, gpu_num_chunks * sizeof(int), cudaMemcpyHostToDevice);
        free(relative_offsets);

        // copy lengths directly since they are already per chunk
        cudaMemcpy(gpu_states[i].d_lengths, lengths + gpu_start_chunk, gpu_num_chunks * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    cudaSetDevice(0); // Reset to the first GPU after allocation
}

extern "C" void get_data(int* data, int* lenghts_out) {
    
    for (int i = 0; i < number_of_gpus; ++i) {
        if (gpu_states[i].d_data == nullptr) continue;

        int gpu_num_chunks = gpu_states[i].end_chunk - gpu_states[i].start_chunk;
        cudaSetDevice(i);
        cudaMemcpy(lenghts_out + gpu_states[i].start_chunk, gpu_states[i].d_lengths, gpu_num_chunks * sizeof(int), cudaMemcpyDeviceToHost);

        int element_counts = gpu_states[i].num_elements;
        cudaMemcpy(data + gpu_states[i].element_start, gpu_states[i].d_data, element_counts * sizeof(int), cudaMemcpyDeviceToHost);
    }
}

extern "C" void get_offsets(int* offsets){
}

extern "C" void get_lengths(int* lengths) {
}

extern "C" void free_elemets() {
    // check if gpu_states is allocated and free the pairs_counters for each GPU
    if (gpu_states != nullptr) {
        for (int i = 0; i < number_of_gpus; ++i) {
            cudaSetDevice(i);
            
            // Free each pointer independently
            if (gpu_states[i].d_pairs_counter != nullptr) {
                printf("Freeing d_pairs_counter for GPU %d\n", i);
                cudaFree(gpu_states[i].d_pairs_counter);
                gpu_states[i].d_pairs_counter = nullptr;
            }
            
            if (gpu_states[i].d_data != nullptr) {
                printf("Freeing d_data for GPU %d\n", i);
                cudaFree(gpu_states[i].d_data);
                gpu_states[i].d_data = nullptr;
            }
            
            if (gpu_states[i].d_offsets != nullptr) {
                printf("Freeing d_offsets for GPU %d\n", i);
                cudaFree(gpu_states[i].d_offsets);
                gpu_states[i].d_offsets = nullptr;
            }
            
            if (gpu_states[i].d_lengths != nullptr) {
                printf("Freeing d_lengths for GPU %d\n", i);
                cudaFree(gpu_states[i].d_lengths);
                gpu_states[i].d_lengths = nullptr;
            }
        }
        
        free(gpu_states);
        printf("Freed gpu_states\n");
        gpu_states = nullptr;
    }
    cudaSetDevice(0); // Reset to the first GPU after allocation
}




__global__ void count_pair_frequencies_kernel(
    int* data,       // Flattened list of all bites
    int* offsets,    // Start of each chunk
    int* lengths,    // Length of each chunk
    int num_chunks, // Number of chunks
    pair_int* pair_counts // Size: MAX_PAIR_KEY
) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d processing chunk %d (GridDim (%d %d), BlockDim (%d %d))\n", threadIdx.x, chunk_id, gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    if (chunk_id >= num_chunks) {
        atomicAdd(&pair_counts[chunk_id], -1.0*chunk_id);
        return;
    }

    int offset = offsets[chunk_id];
    int length = lengths[chunk_id];


    for (int i = 0; i < length - 1; ++i) {
        int a = data[offset + i];
        int b = data[offset + i + 1];
        // if (a < 0 || b < 0) {
        //     break; // Skip negative values
        // }
        int key = (a << BIT_OFFSET) | b;  // Flatten (a,b) into single int key

        atomicAdd(&pair_counts[key], 1);
    }
}

void count_pair_frequencies(int* max_pair, // Max pair to replace with new value
                            int* frequency // Frequency of the max pair
) {

    if (number_of_gpus == 0) {
        fprintf(stderr, "No CUDA devices found. Cannot count pair frequencies.\n");
        return;
    }
    
    std::vector<cudaStream_t> streams(number_of_gpus);
    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaMemsetAsync(gpu_states[i].d_pairs_counter, 0, MAX_PAIR_KEY * sizeof(pair_int), streams[i]);
    }

    for (int i = 0; i < number_of_gpus; ++i) {
        int gpu_num_chunks = gpu_states[i].end_chunk - gpu_states[i].start_chunk;
        if (gpu_num_chunks <= 0) continue;
        int threadsPerBlock = 256;
        int blocksPerGrid = (gpu_num_chunks + threadsPerBlock - 1) / threadsPerBlock;
        // std::cout << "Launching kernel with\n" ;

        count_pair_frequencies_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(gpu_states[i].d_data, gpu_states[i].d_offsets, gpu_states[i].d_lengths, gpu_num_chunks, gpu_states[i].d_pairs_counter);

    }

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error on GPU %d: %s\n", i, cudaGetErrorString(err));
        }
    }

    // Updtating a way ot find the global max pair, from absolute max pair for all GPUs to finding the max pair for each GPU and then comparing those max pairs to find the global max pair. This is more efficient than copying the entire pairs_counter array back to the host for each GPU.

    int global_max_frequency = -1;
     
    for (int i=0; i < number_of_gpus; ++i) {
        if (gpu_states[i].num_elements <= 0) continue;

        cudaSetDevice(i);
        thrust::device_ptr<pair_int> d_counter_ptr(gpu_states[i].d_pairs_counter);
        auto max_iter = thrust::max_element(thrust::device, d_counter_ptr, d_counter_ptr + MAX_PAIR_KEY);

        pair_int local_max_idx = max_iter - d_counter_ptr;
        pair_int local_max_val;
        cudaMemcpy(&local_max_val, thrust::raw_pointer_cast(max_iter),
                   sizeof(pair_int), cudaMemcpyDeviceToHost);

        if (local_max_val > global_max_frequency) {
        
            global_max_frequency = local_max_val;
            max_pair[0] = local_max_idx >> BIT_OFFSET;  // Extract first token
            max_pair[1] = local_max_idx & BIT_MASK;     // Extract second token
            *frequency = local_max_val;
        }
    }

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }

    cudaSetDevice(0);
}

__global__ void compact_kernel(int* data, const int* offsets, int* lengths, const int num_chunks) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks) {
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
    const int num_chunks, // Number of chunks
    int* pair, // Pair to replace
    int* new_value // New value to replace with
) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("Thread %d processing chunk %d (GridDim (%d %d), BlockDim (%d %d)) %d\n", threadIdx.x, chunk_id, gridDim.x, gridDim.y, blockDim.x, blockDim.y, *num_chunks);

    if (chunk_id >= num_chunks) {
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

void replace_single_most_frequent(int* pair, // Pair to replace
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
    

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);

        int num_chunks_for_gpu = gpu_states[i].end_chunk - gpu_states[i].start_chunk;

        if (num_chunks_for_gpu == 0) continue;

        int *d_pair, *d_new_value;

        cudaMalloc(&d_pair, 2 * sizeof(int));
        cudaMalloc(&d_new_value, sizeof(int));
        // cudaMalloc(&num_chunks_gpu, sizeof(int));

        cudaMemcpy(d_pair, pair, 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_value, &new_value, sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(num_chunks_gpu, &num_chunks_for_gpu, sizeof(int), cudaMemcpyHostToDevice);


        int threadsPerBlock = 256;
        int blocksPerGrid = (num_chunks_for_gpu + threadsPerBlock - 1) / threadsPerBlock;

        replace_single_most_frequent_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(gpu_states[i].d_data, gpu_states[i].d_offsets, gpu_states[i].d_lengths, num_chunks_for_gpu, d_pair, d_new_value);
        compact_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(gpu_states[i].d_data, gpu_states[i].d_offsets, gpu_states[i].d_lengths, num_chunks_for_gpu);

        cudaFree(d_pair);
        cudaFree(d_new_value);
        // cudaFree(num_chunks_gpu);
    }

    for (int i = 0; i < number_of_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);

        cudaStreamDestroy(streams[i]);
    }

    cudaSetDevice(0);

}


void load_txt_input(const std::string& filename,
                    std::vector<int>& flat_data,
                    std::vector<int>& offsets,
                    std::vector<int>& lengths) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    int offset = 0;

    while (std::getline(file, line)) {
        // Strip the outer brackets: [[...]] -> [...], [...]
        if (line.size() >= 2 && line.front() == '[' && line.back() == ']') {
            line = line.substr(1, line.size() - 2);
        }

        // Split into sub-arrays by "], ["
        std::vector<std::string> subarrays;
        std::string delimiter = "], [";
        size_t pos = 0;
        while ((pos = line.find(delimiter)) != std::string::npos) {
            subarrays.push_back(line.substr(0, pos));
            line = line.substr(pos + delimiter.size());
        }
        subarrays.push_back(line); // last remaining segment

        for (const std::string& sub : subarrays) {
            int count = 0;
            offsets.push_back(offset);

            // Remove any leftover brackets and parse comma-separated ints
            std::string clean;
            for (char c : sub) {
                if (c != '[' && c != ']') clean += c;
            }

            std::istringstream iss(clean);
            std::string token;
            while (std::getline(iss, token, ',')) {
                // Trim whitespace
                size_t start = token.find_first_not_of(" \t");
                size_t end   = token.find_last_not_of(" \t");
                if (start == std::string::npos) continue;
                token = token.substr(start, end - start + 1);

                if (!token.empty()) {
                    int value = std::stoi(token);
                    flat_data.push_back(value);
                    count++;
                }
            }

            lengths.push_back(count);
            offset += count;
        }
    }
}


int main(int argc, char *argv[]) {
    std::cout << "You have entered " << argc << " arguments:" << std::endl;

    std::string filename;
    if (argc > 1) {
        filename = argv[1];
        std::cout << "Using provided filename: " << filename << std::endl;
    } else {
        filename = "test_input_redux_large.txt";
        std::cout << "No additional arguments provided." << std::endl;
    }

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    std::cout << "Total GPUs: " << gpu_count << std::endl;

    std::vector<int> host_data, host_offsets, host_lengths;
    load_txt_input(filename, host_data, host_offsets, host_lengths);

    std::cout << "\nLoaded data, allocating on GPU(s)...\n";
    printf("Total elements: %d, Total chunks: %d\n", (int)host_data.size(), (int)host_offsets.size());
    printf("First 20 elements: ");
    for (size_t i = 0; i < std::min(host_data.size(), size_t(20)); ++i) {
        std::cout << host_data[i] << " ";
    }
    printf("\n");
    printf("First 20 offsets: ");
    for (size_t i = 0; i < std::min(host_offsets.size(), size_t(20)); ++i) {
        std::cout << host_offsets[i] << " ";
    }
    printf("\n");
    printf("First 20 host_lengths: ");
    for (size_t i = 0; i < std::min(host_lengths.size(), size_t(20)); ++i) {
        std::cout << host_lengths[i] << " ";
    }
    printf("\n");
    allocate_elemets(
        host_data.data(),
        host_offsets.data(),
        host_lengths.data(),
        (int)host_data.size(),
        (int)host_offsets.size()
    );

    std::vector<int> most_frequent_pair = {0, 0};
    int max_frequency = 0;

    count_pair_frequencies(most_frequent_pair.data(), &max_frequency);

    std::cout << "Most frequent pair: ("
              << most_frequent_pair[0] << ", "
              << most_frequent_pair[1]
              << ") with frequency " << max_frequency << "\n";

    std::cout << "Modified IDID:\n";
    int new_token_id = 257; // or whatever your next token ID logic is
    replace_single_most_frequent(most_frequent_pair.data(), new_token_id);

    count_pair_frequencies(most_frequent_pair.data(), &max_frequency);

    std::cout << "Most frequent pair: ("
              << most_frequent_pair[0] << ", "
              << most_frequent_pair[1]
              << ") with frequency " << max_frequency << "\n";

    std::cout << "Modified IDID:\n";
    new_token_id = 258; // or whatever your next token ID logic is
    replace_single_most_frequent(most_frequent_pair.data(), new_token_id);

    // Copy results back from GPU
    std::vector<int> out_data(host_data.size());
    std::vector<int> out_lengths(host_lengths.size());
    get_data(out_data.data(), out_lengths.data());

    std::cout << "Modified Data:\n";
    for (size_t i = 0; i < std::min(out_lengths.size(), size_t(20)); ++i) {
        std::cout << "List " << i << ": ";
        for (int j = 0; j < out_lengths[i]; ++j) {
            std::cout << out_data[host_offsets[i] + j] << " ";
        }
        std::cout << "\n";
    }

    free_elemets();
    return 0;
}