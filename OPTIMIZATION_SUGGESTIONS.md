# BPE Tokenizer Optimization Guide

## Table of Contents

1. [How BPE Training Works (Theory)](#how-bpe-training-works-theory)
2. [Current Architecture Walkthrough](#current-architecture-walkthrough)
3. [GPU Programming Primer](#gpu-programming-primer)
4. [Optimization 1: Keep Data GPU-Resident](#optimization-1-keep-data-gpu-resident)
5. [Optimization 2: Sort+Reduce vs Atomic Hash Table](#optimization-2-sortreduce-vs-atomic-hash-table)
6. [Optimization 3: Eliminate Padding Waste](#optimization-3-eliminate-padding-waste)
7. [Optimization 4: GPU-Side Argmax](#optimization-4-gpu-side-argmax)
8. [Optimization 5: Fast Encode with Priority Queue](#optimization-5-fast-encode-with-priority-queue)
9. [Optimization 6: Batch Independent Merges](#optimization-6-batch-independent-merges)
10. [Optimization 7: Streaming for Large Datasets](#optimization-7-streaming-for-large-datasets)
11. [Optimization 8: Dynamic Memory Management](#optimization-8-dynamic-memory-management)
12. [Optimization 9: Unified Memory](#optimization-9-unified-memory)
13. [Optimization 10: Multi-GPU Load Balancing](#optimization-10-multi-gpu-load-balancing)
14. [Optimization 11: Kernel Fusion](#optimization-11-kernel-fusion)
15. [Optimization 12: Python-Level Improvements](#optimization-12-python-level-improvements)
16. [Priority & Impact Summary](#priority--impact-summary)
17. [Recommended Implementation Roadmap](#recommended-implementation-roadmap)

---

## How BPE Training Works (Theory)

### The Core Algorithm

Byte-Pair Encoding (BPE) is an iterative compression algorithm. Starting from raw bytes (IDs 0-255), it repeatedly finds the most frequent adjacent pair of tokens and fuses them into a new single token.

```
INITIAL STATE (after regex splitting into "words"):
  Word 1: [72, 101, 108, 108, 111]     # "Hello" in UTF-8 bytes
  Word 2: [87, 111, 114, 108, 100]      # "World"

ITERATION 1: Count all adjacent pairs
  (72,101): 1   (101,108): 1   (108,108): 1   (108,111): 2
  (87,111): 1   (111,114): 1   (114,108): 1   (111,100): 1
  → Most frequent: (108,111) with freq 2 → merge into token 256

AFTER MERGE: Replace all (108,111) with 256
  Word 1: [72, 101, 108, 256]           # "Hel" + "lo"
  Word 2: [87, 111, 114, 108, 100]

ITERATION 2: Count pairs again
  (72,101): 1   (101,108): 1   (108,256): 1
  (87,111): 1   (111,114): 1   (114,108): 1   (108,100): 1
  → All freq=1, so merge (72,101) → token 257 (or any tiebreaker)
```

This repeats until the vocabulary reaches the target size or no pair has frequency > 1.

### Why Regex Splitting Matters

The GPT-4 pattern `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+'` splits text into linguistically meaningful units before BPE. This is critical because:

1. **BPE never merges across word boundaries.** The pair `("th", "e")` in "the cat" would never merge across the space, because the regex has already separated them into different chunks.
2. **Each chunk is a self-contained sequence** — the pair counting inside a chunk never interacts with other chunks. This is the property that enables parallelism in the CUDA kernels: one thread per chunk, no inter-chunk communication needed.

### The Data Structure: Chunks

After regex splitting and UTF-8 encoding, your data is a collection of variable-length integer sequences (chunks). This is the heart of the GPU design question:

```
Chunk 0: [72, 101, 108, 108, 111]                   len=5
Chunk 1: [32]                                         len=1
Chunk 2: [87, 111, 114, 108, 100]                    len=5
Chunk 3: [33]                                         len=1
...
Chunk N: [104, 101, 108, 108, 111, 119, 111, 114]    len=8
```

Total elements: ~1 billion (for 1GB text) spread across ~100-500 million chunks (most chunks are short — spaces, punctuation, short words).

---

## Current Architecture Walkthrough

### The Training Loop

```python
# tokenizer.py, __train_cuda__ method (simplified)
ids = text → regex_split → UTF-8 encode → pad to max_length → numpy 2D array

for i in range(vocab_size - 256):
    ids, pair, freq = self.__replace_most_frequent_cuda__(ids, 256 + i)
    # ^^ This single Python call does:
    #   1. __most_frequent_pair_cuda__(ids)       → CUDA: count pairs, find max
    #   2. cuda_replace_single_most_frequent(...)  → CUDA: replace pair, compact
    self.__merges__[pair] = 256 + i
```

### What Happens on the GPU (Per Iteration)

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: H2D COPY                                           │
│  Copy entire padded numpy array from CPU → GPU              │
│  Size: num_chunks × max_length × 4 bytes                    │
│  Example: 10M chunks × 100 max_len × 4B = 4GB               │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: COUNT PAIRS KERNEL                                 │
│  Each thread processes one chunk                            │
│  For each adjacent pair (a,b):                              │
│    key = (a << 14) | b                                      │
│    atomicAdd(&global_counter[key], 1)                       │
│  Counter size: 268,435,456 slots × 4 bytes = 1GB            │
├─────────────────────────────────────────────────────────────┤
│  STEP 3: D2H COPY + CPU SCAN                                │
│  Copy 1GB counter from GPU → CPU                            │
│  Iterate over 268M entries to find the maximum              │
│  Extract pair from key: pair[0] = max_idx >> 14             │
│                         pair[1] = max_idx & 0x3FFF           │
├─────────────────────────────────────────────────────────────┤
│  STEP 4: H2D COPY AGAIN                                     │
│  Copy entire padded array from CPU → GPU (again!)           │
│  Also copy pair and new_value                               │
├─────────────────────────────────────────────────────────────┤
│  STEP 5: REPLACE + COMPACT KERNEL                           │
│  replace: find occurrences of pair, mark with -1            │
│  compact: remove -1 entries within each chunk               │
├─────────────────────────────────────────────────────────────┤
│  STEP 6: D2H COPY AGAIN                                     │
│  Copy modified data from GPU → CPU                          │
└─────────────────────────────────────────────────────────────┘
```

This entire sequence repeats **for every single merge** (e.g., 50,000 times for a 50K-vocab tokenizer). The total data transferred is:

```
Per iteration:
  H2D: 4GB (step 1) + 4GB (step 4) = 8GB
  D2H: 1GB (step 3) + 4GB (step 6) = 5GB
  Total: ~13GB per merge

For 50,000 merges: 650,000 GB transferred across the PCIe bus
At PCIe 4.0 x16 (~32 GB/s theoretical): ~5.6 hours of pure transfer time
```

This is why even though the GPU kernels themselves are fast, the overall training is slow. The PCIe bus is the bottleneck, not compute.

### The 14-Bit Pair Encoding

The current code packs two token IDs into one integer key:

```cuda
#define BIT_OFFSET 14
#define BIT_MASK 0x3FFF     // 16383
#define MAX_PAIR_KEY 268435456  // 16384 × 16384

int key = (a << 14) | b;   // a and b are each up to 14 bits (0-16383)
```

This means the code assumes token IDs will never exceed 16383. For a vocab_size of 1000 this is fine, but for a 50K vocabulary it silently corrupts (50K² pairs need more than 14 bits per token). The pair `(16385, 0)` would be encoded the same as `(1, 0)` because the upper bits are truncated.

---

## GPU Programming Primer

If you're presenting this to developers unfamiliar with CUDA, here are the key concepts:

### The GPU Execution Model

```
CPU (Host)                          GPU (Device)
─────────                           ────────────
                                    ┌─────────────────────┐
 RAM ←──PCIe Bus──→ VRAM            │  SM  SM  SM  SM     │
 (DDR5)   (~32GB/s) (HBM3)         │  SM  SM  SM  SM     │
                                    │    ...               │
                                    └─────────────────────┘
```

- **Host (CPU):** Runs your Python code, manages data, calls CUDA functions.
- **Device (GPU):** Executes kernels in parallel across thousands of cores.
- **Kernel:** A function that runs on the GPU, launched with `<<<grid, block>>>` syntax.
- **Thread:** The smallest unit of execution. Each thread runs the kernel independently.
- **Block:** A group of threads (up to 1024) that can share fast on-chip memory.
- **Grid:** A group of blocks that together process the entire problem.
- **SM (Streaming Multiprocessor):** Hardware unit that executes blocks. An A100 has 108 SMs.

### Memory Hierarchy

```
Register (per thread)     ← Fastest (~0 cycles), very limited (255 per thread)
Shared Memory (per block) ← ~20-30 cycles, 48-164KB per block
L1/L2 Cache               ← Automatic, managed by hardware
Global Memory (VRAM)      ← ~300-800 cycles, gigabytes, accessible by all threads
Pinned Host Memory        ← CPU RAM accessible by GPU DMA, avoids extra copy
Pageable Host Memory      ← Normal CPU RAM, requires staging through pinned buffer
```

### Why Transfers Are Slow

The PCIe bus connecting CPU and GPU is the narrowest pipe in the system:

```
Bandwidth comparison (approximate):
  GPU VRAM (HBM3):       2,000 - 3,300 GB/s
  GPU VRAM (GDDR6X):       700 - 1,000 GB/s
  NVLink (GPU-to-GPU):     600 - 900 GB/s
  PCIe 5.0 x16:             ~63 GB/s
  PCIe 4.0 x16:             ~32 GB/s  ← your likely bottleneck
  PCIe 3.0 x16:             ~16 GB/s
```

Every byte that crosses the PCIe bus costs ~30-100x more than accessing VRAM. The golden rule of GPU programming: **minimize host-device transfers.**

### Why Atomic Operations Are Slow

```cuda
atomicAdd(&global_counter[key], 1);
```

When multiple threads try to write to the same memory address simultaneously:

```
Thread 0:  Read counter[42] → 5
Thread 1:  Read counter[42] → 5   ← collision! Both see same value
Thread 0:  Write counter[42] = 6
Thread 1:  Write counter[42] = 6  ← one increment is lost!
```

`atomicAdd` prevents this by serializing: thread 1 must wait for thread 0 to complete before it can read-modify-write. Under high contention (common pairs in your text like `(32, 32)` for double spaces, or `(101, 32)` for "e " in English), hundreds of threads queue up on the same address, effectively serializing the kernel and destroying the GPU's parallelism advantage.

```
Contention illustration for a common pair like (101, 116) = "et":

Thread 345: waiting on address counter[1653988] ──┐
Thread 1289: waiting on address counter[1653988] ──┤
Thread 4501: waiting on address counter[1653988] ──┤ All serialized
Thread 6723: waiting on address counter[1653988] ──┤ on one address
Thread 8901: waiting on address counter[1653988] ──┘
```

---

## Optimization 1: Keep Data GPU-Resident

### The Problem, Quantified

For each merge iteration, the current code does:

```
C++ function count_pair_frequencies():
  1. cudaMalloc d_data, d_offsets, d_lengths  ← allocate
  2. cudaMemcpyAsync H2D for all arrays        ← copy data to GPU
  3. cudaMemsetAsync counter to 0              ← reset 1GB counter
  4. Launch kernel
  5. cudaMemcpy D2H 1GB counter                ← copy results back
  6. CPU scan for max
  7. cudaFree everything                       ← free GPU memory

C++ function replace_single_most_frequent():
  1. cudaMalloc d_data, d_offsets, d_lengths   ← allocate again
  2. cudaMemcpyAsync H2D for all arrays        ← copy data again
  3. Launch replace kernel + compact kernel
  4. cudaMemcpy D2H result                     ← copy result back
  5. cudaFree everything                       ← free again
```

Every call allocates, copies, and frees. The data lives on the GPU for milliseconds and is then discarded.

### The Fix: Persistent GPU State

```python
# Revised training loop — data stays on GPU for the entire training run
class CUDABPETrainer:
    def __init__(self, flat_data, offsets, lengths, vocab_size):
        self.d_data = cuda_allocate(flat_data)
        self.d_offsets = cuda_allocate(offsets)
        self.d_lengths = cuda_allocate(lengths)
        self.d_counter = cuda_allocate_counter(vocab_size)
        
    def train(self, num_merges):
        for i in range(num_merges):
            # Count runs on GPU-resident data
            pair, freq = self.find_max_pair()
            if freq == 0: break
            
            # Replace also runs on GPU-resident data
            self.replace_and_compact(pair, 256 + i)
            self.merges[pair] = 256 + i
        
        # Pull data back to host only once, at the very end
        self.final_data = cuda_copy_to_host(self.d_data)
        
    def __del__(self):
        cuda_free(self.d_data, self.d_offsets, self.d_lengths, self.d_counter)
```

### Implementation Details

The key change in the CUDA code: separate the counting kernel from the memory management.

```cuda
// BEFORE (current): function allocates, copies, runs, frees
void count_pair_frequencies(int* host_data, int* host_offsets, ...) {
    cudaMalloc(&d_data, ...);
    cudaMemcpy(d_data, host_data, ...);  // H2D every call
    kernel<<<...>>>(d_data, ...);
    cudaMemcpy(host_result, d_counter, ...); // D2H every call
    cudaFree(d_data);
}

// AFTER: data is already on device, only counter is reset
void count_pair_frequencies(int* d_data, int* d_offsets, 
                             int* d_lengths, int num_chunks,
                             pair_int* d_counter,
                             int* max_pair, int* frequency) {
    cudaMemsetAsync(d_counter, 0, counter_size);  // only reset counter
    kernel<<<...>>>(d_data, d_offsets, d_lengths, num_chunks, d_counter);
    // Find max on GPU (see Optimization 4)
}
```

### Why This Is Transformative

```
Before:  13GB transfer per merge × 50,000 merges = 650,000 GB
After:   4GB initial H2D + 4GB final D2H = 8 GB total

Speedup on transfers alone: ~80,000× reduction in PCIe traffic
```

In practice you don't get 80,000x because the kernels themselves take time, but you eliminate the dominant bottleneck entirely.

---

## Optimization 2: Sort+Reduce vs Atomic Hash Table

### Theory: Why Sort+Reduce Is Better for BPE

The atomic hash table approach is essentially a **scatter** operation: each thread scatters its pairs into a giant array using atomic updates. Sort+reduce is a **gather** approach:

#### Hash Table (Current)
```
Input pairs:  [(108,108), (108,111), (111,32), (32,119), (108,108), ...]
                   │           │         │        │         │
                   ▼           ▼         ▼        ▼         ▼
           atomicAdd  atomicAdd  atomicAdd atomicAdd atomicAdd
                   │           │         │        │         │
                   └───────────┴─────────┴────────┴─────────┘
                                       │
                                       ▼
                        Giant counter array (268M slots)
                        Only ~100K slots are non-zero
                                       │
                                       ▼
                        CPU scan over 268M to find max
```

#### Sort+Reduce (Proposed)
```
Input pairs:  [(108,108), (108,111), (111,32), (32,119), (108,108), ...]
                    │
                    ▼
             Sort the pairs  ← O(N log N), highly parallel on GPU
                    │
                    ▼
    Sorted:   [(32,119), (108,108), (108,108), (108,111), (111,32), ...]
                    │
                    ▼
           Reduce by key    ← O(N), neighbors with same key get counted
                    │
                    ▼
    Compressed: [((32,119), 1), ((108,108), 2), ((108,111), 1), ((111,32), 1)]
                    │
                    ▼
              max_element   ← O(K) where K = unique pairs, on GPU
                    │
                    ▼
            Result: ((108,108), 2)
```

This is the algorithm used by production GPU BPE implementations (HuggingFace Tokenizers, NVIDIA's cuDF BPE, YouTokenToMe).

### Why It Avoids Contention

There are no atomic operations in the sort+reduce pipeline. Radix sort (which Thrust uses for integer keys) runs in parallel with each thread operating on independent data. The reduce step compares adjacent elements — again independent. No two threads ever write to the same address simultaneously.

### Step-by-Step Implementation

**Step 1: Extract pairs**

```cuda
__global__ void extract_pairs_kernel(
    const int* data,
    const int* offsets,
    const int* lengths,
    int num_chunks,
    unsigned long long* pairs_out,  // (token_a << 32) | token_b
    int* pair_count_out             // number of pairs extracted
) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks) return;
    
    int offset = offsets[chunk_id];
    int len = lengths[chunk_id];
    
    // Each thread writes its pairs to a pre-computed offset in the output buffer
    // (need a prefix-sum of (lengths - 1) beforehand to know where)
    int write_offset = chunk_pair_offsets[chunk_id];
    
    for (int i = 0; i < len - 1; i++) {
        int a = data[offset + i];
        int b = data[offset + i + 1];
        pairs_out[write_offset + i] = ((unsigned long long)a << 32) | (unsigned int)b;
    }
}
```

**Step 2-4: Sort, reduce, find max**

```c++
// Thrust does the heavy lifting
thrust::device_ptr<unsigned long long> d_pairs(pairs_out);
thrust::device_ptr<int> d_counts(counts_out);

// Sort: O(N log N) with highly tuned radix sort
thrust::sort(thrust::device, d_pairs, d_pairs + total_pairs);

// Reduce by key: compresses runs of identical keys
auto new_end = thrust::reduce_by_key(
    thrust::device,
    d_pairs, d_pairs + total_pairs,         // input keys
    thrust::constant_iterator<int>(1),       // input values (all 1s)
    d_unique_pairs,                          // output unique keys
    d_counts                                 // output counts
);

int num_unique = new_end.first - d_unique_pairs;

// Find max: O(num_unique)
auto max_iter = thrust::max_element(thrust::device, d_counts, d_counts + num_unique);
int max_idx = max_iter - d_counts;
unsigned long long max_pair = d_unique_pairs[max_idx];
int max_count = d_counts[max_idx];
```

**Step 5: Decode the pair**

```c++
int token_a = max_pair >> 32;
int token_b = max_pair & 0xFFFFFFFF;
```

This uses 64-bit keys, supporting token IDs up to 2^32 (4 billion) — effectively unlimited vocabulary.

### Memory Analysis: Sort+Reduce vs Hash Table

```
// Hash table approach (current):
//   268M slots × 4 bytes = 1,073 MB permanently allocated
//   Memory usage is CONSTANT regardless of data size

// Sort+reduce approach:
//   N pairs × 8 bytes (key) + N pairs × 4 bytes (value stencil)
//   For 10MB text:  ~10M tokens → ~10M pairs → ~120 MB
//   For 100MB text: ~100M tokens → ~100M pairs → ~1.2 GB
//   For 1GB text:   ~1B tokens → ~1B pairs → ~12 GB (may need batching)
//   Memory usage is PROPORTIONAL to data size
```

For the 1GB case where 12GB might not fit, you can batch:

```python
def count_pairs_batched(d_data, d_offsets, d_lengths, num_chunks, batch_size=100000):
    """Process chunks in batches to limit GPU memory."""
    merged_histogram = {}  # or use thrust::sort + thrust::merge on GPU
    
    for start in range(0, num_chunks, batch_size):
        end = min(start + batch_size, num_chunks)
        batch_chunks = end - start
        
        # Extract, sort, reduce this batch
        pairs, counts = extract_sort_reduce_batch(
            d_data, d_offsets + start, d_lengths + start, batch_chunks
        )
        # Merge histogram (can be done on GPU with thrust::merge)
        merged_histogram = merge_histograms(merged_histogram, pairs, counts)
    
    return find_max(merged_histogram)
```

### Real-World Performance Reference

NVIDIA's cuDF implementation of BPE using sort+reduce achieves ~100-500 million pairs/second on an A100. At that rate, counting pairs in a 1GB dataset takes <10 seconds, even with batching.

---

## Optimization 3: Eliminate Padding Waste

### The Waste, Quantified

The GPT-4 regex pattern produces chunks with a power-law distribution. Here's a realistic distribution from English text:

```
Chunk length distribution for 1GB of English text:
  Length 1:   ████████████████████████████████  40% (spaces, punctuation)
  Length 2-3: ████████████████                  20% (short words: "a", "is", "the")
  Length 4-8: ████████████████                  30% (common words)
  Length 9-20: ████                              8% (longer words)
  Length 20+:  ██                                2% (long words, URLs, code spans)
```

With padding to max_length (let's say 100):

```
Memory used:       num_chunks × max_length × 4 bytes
Actual data:       sum(actual_lengths) × 4 bytes
Waste:             (num_chunks × max_length - sum(actual_lengths)) × 4 bytes

For 1GB text, ~200M chunks, avg length 5, max length 100:
  Memory used: 200M × 100 × 4 = 80 GB  ← likely doesn't fit
  Actual data: 200M × 5 × 4   = 4 GB
  Waste: 76 GB (95%!)
```

### The Fix: CSR-Style Flat Storage

```
BEFORE (padded dense, 2D array):
  [[72, 101, 108, 108, 111, -1, -1, ..., -1],    ← 100 elements
   [32, -1, -1, ..., -1],                          ← 100 elements
   [87, 111, 114, 108, 100, -1, ..., -1],         ← 100 elements
   ...]

AFTER (flat array + metadata, CSR format):
  data:    [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, ...]
           ◄──── chunk 0 ────► ◄c1► ◄─────── chunk 2 ──────────►

  offsets: [0, 5, 6, 11, ...]     ← offset[i] = start of chunk i in data[]
  lengths: [5, 1, 5, ...]         ← lengths[i] = number of tokens in chunk i
```

This is the format your CUDA kernels already expect (see `count_pair_frequencies_kernel` parameters). The Python side just isn't supplying data in this format yet.

### Python Implementation

```python
def __train_cuda__(self, text, vocab_size, verbose=False):
    text_chunks = re.findall(self.tiktoken_pat, text)
    ids = [list(ch.encode("utf-8")) for ch in text_chunks]
    
    # Build CSR representation
    offsets = np.zeros(len(ids) + 1, dtype=np.int32)
    lengths = np.array([len(row) for row in ids], dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    
    total_elements = offsets[-1]
    flat_data = np.empty(total_elements, dtype=np.int32)
    for i, row in enumerate(ids):
        flat_data[offsets[i]:offsets[i+1]] = row
    
    # Now pass flat_data, offsets, lengths to CUDA
    # No padding, no -1 sentinel, no wasted memory
```

### GPU Kernel Simplification

Without padding, the kernel no longer needs to check for `-1` sentinel values:

```cuda
// BEFORE: must check for -1 padding
for (int i = 0; i < length - 1; ++i) {
    int a = data[offset + i];
    int b = data[offset + i + 1];
    if (a < 0 || b < 0) break;  // sentinel check
    // ...
}

// AFTER: all data is real, no checks needed
for (int i = 0; i < length - 1; ++i) {
    int a = data[offset + i];
    int b = data[offset + i + 1];
    // ...
}
```

This removes a conditional branch from the innermost loop, which matters for GPU performance (branch divergence within a warp costs execution time).

---

## Optimization 4: GPU-Side Argmax

### The Problem

Currently, the 1GB counter array is copied to the CPU and scanned there. This is wrong on two levels:

1. **Transfer cost:** 1GB over PCIe every iteration
2. **CPU work:** Scanning 268M ints on the CPU is slow (~50-100ms per scan on a fast CPU)

### The Fix: GPU Reduction

After sort+reduce, the unique pair array is much smaller (hundreds of thousands, not hundreds of millions) and already on the GPU. `thrust::max_element` finds the maximum in microseconds:

```cuda
// After reduce_by_key, result is on GPU:
// d_unique_pairs: [pair1, pair2, pair3, ...]  (num_unique elements)
// d_counts:       [5,      2,      99,    ...]  (num_unique elements)

auto max_iter = thrust::max_element(
    thrust::device, 
    d_counts, 
    d_counts + num_unique
);

int max_index = max_iter - d_counts;
int max_freq = d_counts[max_index];
unsigned long long max_pair = d_unique_pairs[max_index];

// Transfer only these two values (8 bytes total) to the host
cudaMemcpy(&host_pair, &d_unique_pairs[max_index], sizeof(unsigned long long), 
           cudaMemcpyDeviceToHost);
cudaMemcpy(&host_freq, &d_counts[max_index], sizeof(int),
           cudaMemcpyDeviceToHost);
```

### Multi-GPU: Hierarchical Reduction

For multi-GPU, each GPU finds its local maximum, then only the candidates are compared on the CPU:

```
GPU 0: local_max = ((108, 111), freq=5042)  ─┐
GPU 1: local_max = ((108, 111), freq=4987)  ─┤ transfer 8 bytes each
GPU 2: local_max = ((101, 116), freq=5123)  ─┤  → CPU compares 3 candidates
GPU 3: local_max = ((108, 111), freq=5001)  ─┘

CPU: global_max = max(freqs) = ((101, 116), freq=5123)  from GPU 2
```

Data transferred per merge drops from 1GB to ~100 bytes for multi-GPU, or just 8 bytes for single-GPU.

---

## Optimization 5: Fast Encode with Priority Queue

### Why Current Encode Is Slow

`__encode_chunk__` applies merges sequentially:

```python
def __encode_chunk__(self, tokens):
    past_list = list(tokens)
    for pair, idx in self.__merges__.items():    # loops 50,000 times
        i = 0
        new_list = []
        while i < len(past_list) - 1:            # scans entire list each time
            if pair == (past_list[i], past_list[i+1]):
                new_list.append(idx)
                i += 2
            else:
                new_list.append(past_list[i])
                i += 1
        past_list = list(new_list)
    return new_list
```

For one chunk of length 50 with 50,000 merges: 50,000 × 50 = 2.5M operations. For 1M chunks: 2.5 trillion operations. Most merge lookups fail because the merges are ordered by training time, not by which pairs actually exist in the current chunk.

### The Fix: Priority Queue (Huffman-Style)

Instead of iterating through ALL merges, only consider the pairs that actually exist in the current word:

```python
import heapq

def _encode_word(self, word):
    """Encode a single word using min-heap of adjacent pairs."""
    if len(word) < 2:
        return word
    
    # word is a list of ints (byte IDs)
    # Each element in the heap: (merge_priority, position, pair)
    # merge_priority = rank of this pair in self.__merges__ (lower = added earlier)
    # When two pairs have the same priority, earlier ones should merge first
    
    nodes = list(word)
    
    # Build priority queue of all adjacent pairs
    # Priority = merge rank (index in merge order, 0 = first merge, highest priority)
    heap = []
    for i in range(len(nodes) - 1):
        pair = (nodes[i], nodes[i + 1])
        if pair in self.__merge_rank__:  # precomputed: {pair: rank}
            heapq.heappush(heap, (self.__merge_rank__[pair], i, pair))
    
    # While there are merges to apply
    while heap:
        rank, pos, (a, b) = heapq.heappop(heap)
        
        # Check if this position is still valid (neighbors may have been merged)
        if pos >= len(nodes) - 1:
            continue
        if nodes[pos] != a or nodes[pos + 1] != b:
            continue
        
        # Apply merge
        new_id = self.__merges__[(a, b)]
        nodes[pos] = new_id
        del nodes[pos + 1]
        
        # Update heap with new neighbors
        if pos > 0:
            left_pair = (nodes[pos - 1], nodes[pos])
            if left_pair in self.__merge_rank__:
                heapq.heappush(heap, (self.__merge_rank__[left_pair], pos - 1, left_pair))
        if pos < len(nodes) - 1:
            right_pair = (nodes[pos], nodes[pos + 1])
            if right_pair in self.__merge_rank__:
                heapq.heappush(heap, (self.__merge_rank__[right_pair], pos, right_pair))
    
    return nodes
```

This is O(n log n) per word instead of O(n × m). For a 50-token word and 50K merges:
- Before: 50 × 50,000 = 2.5M iterations
- After: ~50 × log(50) ≈ 280 iterations (only actually applicable merges)

### Precomputing the Merge Rank

```python
def __init__(self, ...):
    ...
    self.__merge_rank__ = {}  # (a,b) → rank in merge order
    
def __train_cuda__(self, ...):
    for i in range(num_merges):
        ...
        self.__merges__[pair] = idx
        self.__merge_rank__[pair] = i  # lower = merged earlier = higher priority
```

---

## Optimization 6: Batch Independent Merges

### Theory: Which Merges Can Be Batched?

Two merges `(A,B)→X` and `(C,D)→Y` are **independent** if applying one does not affect the frequency of the other:

```
Independent:     (101, 116) → 300  and  (32, 119) → 301
  ✓ These pairs don't share any token IDs
  ✓ Replacing "et" doesn't affect " w" count

Dependent:        (101, 116) → 300  and  (116, 32) → 301
  ✗ After merging "et" (101,116)→300, the sequence [101, 116, 32]
    becomes [300, 32], so the count for (116, 32) changes
  ✗ Also: [..., 101, 116, 32] → after first merge: [..., 300, 32]
    The pair (116, 32) no longer exists here
```

### Independence Check

```python
def select_independent(pairs_with_freqs, max_pairs=64):
    """
    Greedy selection: take highest-frequency pairs, skip if they conflict
    with any already-selected pair.
    """
    selected = []
    used_tokens = set()
    
    # Sort by frequency descending
    for (a, b), freq in sorted(pairs_with_freqs, key=lambda x: -x[1]):
        # A new pair conflicts if either token appears in an already-selected pair
        # or if either token equals a newly-created token
        new_id = next_available_id()
        if a in used_tokens or b in used_tokens or new_id in used_tokens:
            continue
        
        selected.append(((a, b), new_id, freq))
        used_tokens.update([a, b, new_id])
        
        if len(selected) >= max_pairs:
            break
    
    return selected
```

### Multi-Pair Replace Kernel

```cuda
__global__ void replace_multiple_pairs_kernel(
    int* data,
    const int* offsets,
    const int* lengths,
    int num_chunks,
    const int* pairs_to_replace,  // flat array: [a1,b1,new1, a2,b2,new2, ...]
    int num_pairs
) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks) return;
    
    int offset = offsets[chunk_id];
    int len = lengths[chunk_id];
    
    // Build a small lookup table in shared memory
    __shared__ int lookup[64][3];  // [pair_idx][a, b, new_id]
    if (threadIdx.x < num_pairs) {
        lookup[threadIdx.x][0] = pairs_to_replace[threadIdx.x * 3];
        lookup[threadIdx.x][1] = pairs_to_replace[threadIdx.x * 3 + 1];
        lookup[threadIdx.x][2] = pairs_to_replace[threadIdx.x * 3 + 2];
    }
    __syncthreads();
    
    for (int i = 0; i < len - 1; i++) {
        int a = data[offset + i];
        int b = data[offset + i + 1];
        
        for (int p = 0; p < num_pairs; p++) {
            if (a == lookup[p][0] && b == lookup[p][1]) {
                data[offset + i] = lookup[p][2];
                data[offset + i + 1] = -1;
                i++;  // skip the consumed token
                break;
            }
        }
    }
}
```

### Batching Effectiveness

In practice, for English text with a 50K-vocab target:
- Early merges: ~20-40 independent pairs per iteration (many byte pairs are independent)
- Late merges: ~5-15 independent pairs (tokens overlap more)
- Average: ~20 pairs per kernel launch → 50,000 / 20 = 2,500 launches instead of 50,000

The counting cost is the same either way (you have to count all pairs regardless), so the saving is in the replace+compact kernel launches and their associated overhead.

---

## Optimization 7: Streaming for Large Datasets

### Why Streaming Works for BPE

BPE pair frequencies are **count-based and order-independent**. The global frequency of pair `(a,b)` is simply the sum of its frequencies across all data shards:

```
freq_global(a,b) = freq_shard_1(a,b) + freq_shard_2(a,b) + ... + freq_shard_N(a,b)
```

This means you can:
1. Count pairs on each shard independently (in parallel)
2. Sum the frequency histograms
3. Find the global maximum
4. Apply the merge across all shards

### Streaming Architecture

```
                          ┌────────────────────┐
 1GB text file ──────────►│  Split into shards  │
                          │  of ~64MB each      │
                          └────────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ Shard 0  │  │ Shard 1  │  │ Shard N  │
              │ (regex   │  │ (regex   │  │ (regex   │
              │  split)  │  │  split)  │  │  split)  │
              └────┬─────┘  └────┬─────┘  └────┬─────┘
                   │             │             │
                   ▼             ▼             ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ Count    │  │ Count    │  │ Count    │
              │ pairs    │  │ pairs    │  │ pairs    │
              │ (GPU)    │  │ (GPU)    │  │ (GPU)    │
              └────┬─────┘  └────┬─────┘  └────┬─────┘
                   │             │             │
                   └─────────────┼─────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Merge histograms │
                        │ (CPU or GPU)     │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Find global max │
                        │ Apply merge to  │
                        │ all shards (GPU)│
                        └─────────────────┘
```

### When to D2H

You only need to pull data back to the host when:
1. Training is complete (final tokenized data)
2. You need to report progress (optional, can be async)
3. The dataset doesn't fit in GPU memory and you need to swap shards

### Overlapping I/O with Compute

```python
def train_streaming(self, file_path, vocab_size, shard_size_mb=64):
    shards = split_file_into_shards(file_path, shard_size_mb)
    
    # Pre-process first shard
    current_shard = preprocess_shard(shards[0])
    gpu_data = copy_to_gpu(current_shard)
    
    for i in range(vocab_size - 256):
        # Count on current shard
        pair, freq = count_pairs_gpu(gpu_data)
        
        # While GPU is counting, pre-load next shard on CPU
        next_shard_idx = (i + 1) % len(shards)
        next_shard = preprocess_async(shards[next_shard_idx])
        
        if freq == 0:
            # Current shard exhausted, swap to next
            gpu_data = copy_to_gpu(next_shard)
            continue
        
        # Apply merge on GPU
        replace_gpu(gpu_data, pair, 256 + i)
        self.__merges__[pair] = 256 + i
```

For a 1GB file split into 16 shards of 64MB, each shard comfortably fits in GPU memory, and the I/O of loading the next shard can be hidden behind GPU computation.

---

## Optimization 8: Dynamic Memory Management

### The Fixed Counter Problem

```cuda
#define MAX_PAIR_KEY 268435456  // 16384² — hardcoded assumption

// At vocab_size=1000: only 1,000,000 possible pairs (0.4% utilization)
//   → 99.6% of the counter is zeros, but you still scan all of it

// At vocab_size=50000: 2,500,000,000 possible pairs (931% overflow)
//   → silent data corruption as pair keys collide
```

The 14-bit encoding `(a << 14) | b` means token `a` must be < 16384. When your vocabulary reaches 16384, any token ID ≥ 16384 gets truncated to 14 bits, causing different pairs to map to the same key.

### Fix: Adaptive Bit Width

If keeping the hash table approach, dynamically grow the encoding:

```cuda
// Find the current max token ID
int max_token_id = *thrust::max_element(thrust::device, d_data, d_data + num_elements);

// Determine bits needed
int bits = 1;
while ((1 << bits) <= max_token_id) bits++;

// Reallocate counter if needed
size_t new_size = (1ULL << (bits * 2));
if (new_size > current_counter_size) {
    cudaFree(d_counter);
    cudaMalloc(&d_counter, new_size * sizeof(pair_int));
    current_counter_size = new_size;
}
```

But as noted in Optimization 2, the sort+reduce approach eliminates this entire problem by not having a fixed-size counter.

---

## Optimization 9: Unified Memory

### What Unified Memory Does

CUDA Unified Memory (`cudaMallocManaged`) creates allocations that are accessible from both CPU and GPU through the same pointer:

```cuda
// Traditional: two separate allocations, explicit copies
int* host_data = malloc(size);
int* device_data;
cudaMalloc(&device_data, size);
cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);  // manual
kernel<<<...>>>(device_data);
cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost);  // manual

// Unified Memory: single allocation, no copies (at least in source code)
int* data;
cudaMallocManaged(&data, size);  // accessible from BOTH CPU and GPU
// ... initialize data on CPU ...
kernel<<<...>>>(data);  // GPU accesses it directly
cudaDeviceSynchronize();
// ... read results on CPU from same pointer ...
```

### How It Works Under the Hood

On modern GPUs (Pascal architecture / CUDA 8+), unified memory uses **page migration**:

1. Initially, pages reside on the CPU
2. When the GPU accesses a page, it triggers a page fault
3. The GPU driver migrates the page to GPU memory on demand
4. When the CPU accesses it again, it migrates back

This is essentially demand-paged virtual memory for GPUs. The key insight is that for our BPE workload, the access pattern is well-defined: CPU initializes → GPU processes repeatedly → CPU reads final result. The data migrates once each way.

### When to Use It, When Not To

```
Good for unified memory:
  ✓ Data accessed primarily by one processor at a time
  ✓ Access patterns are predictable (CPU init → GPU compute → CPU read)
  ✓ Simplifying complex multi-GPU code

Bad for unified memory:
  ✗ CPU and GPU both access the same data simultaneously
  ✗ Fine-grained interleaved access (page fault overhead dominates)
  ✗ Need absolute peak bandwidth (explicit copies can be optimized better)
```

For this BPE tokenizer, if you implement Optimization 1 (data stays on GPU), unified memory is unnecessary for the data arrays — but it's nice for the pair counter and metadata.

---

## Optimization 10: Multi-GPU Load Balancing

### The Current Problem

```c++
int gpu_chunk_size = (num_chunks + number_of_gpus - 1) / number_of_gpus;

// GPU 0 gets chunks 0-249,999
// GPU 1 gets chunks 250,000-499,999
// ...
```

This distributes chunks evenly by **count**, but chunks vary wildly in size:

```
GPU 0: 250,000 chunks × avg 3 tokens = 750,000 tokens   ← underutilized
GPU 1: 250,000 chunks × avg 5 tokens = 1,250,000 tokens
GPU 2: 250,000 chunks × avg 8 tokens = 2,000,000 tokens  ← bottleneck
```

GPU 0 finishes first and sits idle while GPU 2 is still working.

### Fix: Balance by Total Token Count

```c++
// Compute prefix sum of chunk lengths
std::vector<int> prefix_sum(num_chunks + 1, 0);
for (int i = 0; i < num_chunks; i++) {
    prefix_sum[i + 1] = prefix_sum[i] + lengths[i];
}

int total_tokens = prefix_sum[num_chunks];
int tokens_per_gpu = (total_tokens + number_of_gpus - 1) / number_of_gpus;

// Assign chunks to GPUs such that each gets ~tokens_per_gpu tokens
std::vector<int> gpu_chunk_ranges(number_of_gpus + 1, 0);
gpu_chunk_ranges[0] = 0;
int current_gpu = 0;
for (int i = 1; i <= num_chunks && current_gpu < number_of_gpus; i++) {
    if (prefix_sum[i] >= (current_gpu + 1) * tokens_per_gpu) {
        gpu_chunk_ranges[++current_gpu] = i;
    }
}
gpu_chunk_ranges[number_of_gpus] = num_chunks;

// GPU i processes chunks [gpu_chunk_ranges[i], gpu_chunk_ranges[i+1])
```

### Dynamic Scheduling (Work Stealing)

For even better balance, use a global atomic counter:

```cuda
__device__ int global_chunk_counter = 0;

__global__ void count_pairs_dynamic(int* data, int* offsets, int* lengths, 
                                      int num_chunks, pair_int* counter) {
    while (true) {
        int chunk_id = atomicAdd(&global_chunk_counter, 1);
        if (chunk_id >= num_chunks) break;
        
        // Process chunk_id
        int offset = offsets[chunk_id];
        int length = lengths[chunk_id];
        for (int i = 0; i < length - 1; i++) {
            int key = (data[offset + i] << BIT_OFFSET) | data[offset + i + 1];
            atomicAdd(&counter[key], 1);
        }
    }
}
```

This naturally balances load: GPUs that finish faster simply grab more chunks. No idle time.

---

## Optimization 11: Kernel Fusion

### Replace + Compact as One Kernel

Currently:

```cuda
// Kernel 1: Replace pairs with -1 markers
replace_single_most_frequent_kernel<<<...>>>(...);
// Kernel 2: Remove -1 entries, compact chunks
compact_kernel<<<...>>>(...);
```

These can be fused because the compaction only depends on data within the same chunk, and replacement runs chunk-by-chunk:

```cuda
__global__ void replace_and_compact_kernel(
    int* data,
    const int* offsets,
    int* lengths,
    int num_chunks,
    int pair_a, int pair_b,
    int new_value
) {
    extern __shared__ int shared_buffer[];  // per-block scratch space
    
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks) return;
    
    int offset = offsets[chunk_id];
    int len = lengths[chunk_id];
    
    // Step 1: Read chunk into shared memory
    // (one chunk per block, unless chunks are very long)
    int write_pos = 0;
    for (int i = 0; i < len; i++) {
        int a = data[offset + i];
        
        if (i < len - 1) {
            int b = data[offset + i + 1];
            if (a == pair_a && b == pair_b) {
                shared_buffer[write_pos++] = new_value;
                i++;  // skip b
                continue;
            }
        }
        shared_buffer[write_pos++] = a;
    }
    
    // Step 2: Write back compacted chunk
    for (int i = 0; i < write_pos; i++) {
        data[offset + i] = shared_buffer[i];
    }
    lengths[chunk_id] = write_pos;
}
```

The savings here are modest (avoiding one kernel launch overhead and one pass over the data), but it adds up over thousands of merges.

---

## Optimization 12: Python-Level Improvements

### 12a: Encode Parallelism (Use That Pool)

The multiprocessing pool created on line 30 is never used. Encoding is embarrassingly parallel — each chunk is independent:

```python
def encode(self, text):
    text_chunks = re.findall(self.tiktoken_pat, text)
    ids = [list(ch.encode("utf-8")) for ch in text_chunks]
    
    # Distribute chunks across workers
    chunk_batches = self.split_into_chunks(ids, self.optimal_size)
    encoded_batches = self._mpool_.map(
        self._encode_batch, 
        [(batch, self.__merges__) for batch in chunk_batches]
    )
    
    return [token for batch in encoded_batches for token in batch]

@staticmethod
def _encode_batch(args):
    chunks, merges = args
    return [RegexTokenizer.__encode_chunk_static__(chunk, merges) for chunk in chunks]
```

### 12b: Fix `__del__` → `atexit` or Context Manager

`__del__` is called when the object is garbage collected, which may never happen or happen during interpreter shutdown when modules are already unloaded:

```python
# Instead of:
def __del__(self):
    self._mpool_.close()

# Do:
import atexit
atexit.register(self._mpool_.close)

# Or even better, use a context manager:
def __enter__(self):
    return self

def __exit__(self, *args):
    self._mpool_.close()
    self._mpool_.join()
```

### 12c: Move `literal_eval` Import

```python
# Line 177 — import inside function called on every load()
from ast import literal_eval

# Move to top of file, or better, don't use it at all.
# Instead of storing pairs as "(101, 116)" and parsing with literal_eval,
# use a structured format:

# Save:
json.dump({str(k): v for k, v in self.__merges__.items()}, f)

# Load:
self.__merges__ = {tuple(json.loads(k)): v for k, v in loaded.items()}
```

### 12d: Decouple Tokenizer from Training

Currently `__init__` either loads or trains. This couples the tokenizer logic with training logic:

```python
class RegexTokenizer:
    def __init__(self):
        self.__vocab__ = {idx: bytes([idx]) for idx in range(256)}
        self.__merges__ = {}
        self.tiktoken_pat = re.compile(GPT4_SPLIT_PATTERN)
    
    def train(self, text, vocab_size):
        trainer = CUDABPETrainer(text, vocab_size)
        trainer.run()
        self.__merges__ = trainer.merges
        self.__vocab__ = trainer.vocab
    
    @classmethod
    def from_pretrained(cls, path):
        tokenizer = cls()
        tokenizer.load(path)
        return tokenizer
```

---

## Priority & Impact Summary

The improvements are ordered by the combination of impact and effort. The top two (#1 + #2) together transform the training loop.

| # | Change | Category | Effort | Impact | Why |
|---|--------|----------|--------|--------|-----|
| 1 | GPU-resident data | Architecture | Medium | **Massive** | Eliminates ~80,000x PCIe traffic |
| 2 | Sort+reduce counting | Algorithm | Medium | **Massive** | No contention, no fixed counter, GPU argmax |
| 3 | CSR format (no padding) | Data layout | Small | **Large** | 10-100x memory savings, simpler kernels |
| 4 | GPU-side argmax | Algorithm | Small | **Large** | Eliminates 1GB D2H per iteration |
| 5 | Fast encode (priority queue) | Algorithm | Medium | **Large** | 100-1000x faster encoding |
| 6 | Batch independent merges | Algorithm | Medium | **Large** | 5-20x fewer kernel launches |
| 7 | Streaming for large files | Architecture | Medium | **Medium** | Enables >GPU-memory datasets |
| 8 | Dynamic memory | Memory | Small | **Medium** | Safety + efficiency across vocab sizes |
| 9 | Unified memory | Implementation | Small | **Small** | Simpler code, fewer bugs |
| 10 | Multi-GPU load balance | Performance | Small | **Small** | Better utilization |
| 11 | Kernel fusion | Performance | Small | **Small** | Modest kernel speedup |
| 12 | Python cleanups | Code quality | Small | **Small** | Better code, parallel encode |

---

## Recommended Implementation Roadmap

### Phase 1: Fix the Data Pipeline (1-2 days)

Implement optimizations #3 and #1 together, since #3 (CSR format) is a prerequisite for clean GPU-resident data:

1. Replace padded 2D array with flat `data`, `offsets`, `lengths` arrays in Python
2. Refactor CUDA functions to accept persistent device pointers
3. Allocate GPU memory once, free once
4. Verify correctness against the current implementation

**Expected result:** Training works on datasets that previously didn't fit in memory, and PCIe transfers drop to initial + final only.

### Phase 2: Sort+Reduce Counting (2-3 days)

This is the largest single change but gives the most benefit:

1. Implement `extract_pairs_kernel` that writes pairs to an output buffer
2. Use `thrust::sort` + `thrust::reduce_by_key` + `thrust::max_element`
3. Implement batched processing for large datasets (use Phase 1's GPU-resident data)
4. Implement multi-pair batching (#6) on top of this

**Expected result:** 5-50x faster counting per iteration, no fixed memory overhead, GPU-side argmax.

### Phase 3: Encode Optimization (1 day)

1. Implement priority-queue-based encoding (#5)
2. Enable multiprocessing for parallel encode (#12a)
3. Profile encode on realistic text → should be near-instant for typical inputs

### Phase 4: Production Hardening (1-2 days)

1. Stream large files from disk (#7)
2. Unified memory cleanup (#9)
3. Multi-GPU load balancing (#10)
4. Code quality improvements (#12b, #12c, #12d)
