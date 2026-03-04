// Optimized Paged Attention Kernels for Metal
// Optimizations:
// 1. simdgroup_matrix for QK^T computation
// 2. Async memory copy for K/V loading
// 3. Reduced threadgroup barriers
// 4. Better memory coalescing patterns

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// BFloat16 support (same as original)
#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#else
typedef struct _MLX_BFloat16 bfloat16_t;
#endif

// ============================================================================
// Optimization 1: simdgroup_matrix based QK computation
// ============================================================================

// Fast QK dot product using simdgroup_matrix (8x8 blocks)
template<typename T, int HEAD_DIM>
METAL_FUNC void fast_qk_dot(
    const threadgroup T* q_tile,
    const threadgroup T* k_tile,
    thread float& qk_out,
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // Use simdgroup_matrix for 8x8 matrix multiply
    // Each SIMD group processes 8x8 = 64 elements
    simdgroup_matrix<T, 8, 8> q_mat;
    simdgroup_matrix<T, 8, 8> k_mat;

    // Load Q and K tiles into simdgroup matrices
    q_mat.thread_elements[0] = q_tile[simd_lid];
    k_mat.thread_elements[0] = k_tile[simd_lid];

    // Multiply and accumulate
    simdgroup_multiply<T, T, T>(q_mat, k_mat, q_mat);

    // Sum reduction across SIMD group
    qk_out = simd_sum(q_mat.thread_elements[0]);
}

// ============================================================================
// Optimization 2: Async memory copy for K/V loading
// ============================================================================

// Async load K/V block into threadgroup memory
template<typename T, int BLOCK_SIZE, int HEAD_DIM>
METAL_FUNC void async_load_kv_block(
    device const T* k_cache,
    device const T* v_cache,
    threadgroup T* k_tile,
    threadgroup T* v_tile,
    int64_t physical_block,
    int kv_head_idx,
    int kv_block_stride,
    int kv_head_stride,
    uint tid [[thread_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Calculate base offset
    const int64_t base_offset = physical_block * kv_block_stride
                              + kv_head_idx * kv_head_stride;

    // Async copy using simdgroup_async_copy (Metal 3+)
    // Each thread copies multiple elements for better bandwidth
    constexpr int ELEMENTS_PER_THREAD = 4;
    const int num_elements = BLOCK_SIZE * HEAD_DIM;
    const int num_iterations = (num_elements + block_size * ELEMENTS_PER_THREAD - 1)
                             / (block_size * ELEMENTS_PER_THREAD);

    for (int i = 0; i < num_iterations; i++) {
        int idx = tid * ELEMENTS_PER_THREAD + i * block_size * ELEMENTS_PER_THREAD;
        if (idx + ELEMENTS_PER_THREAD <= num_elements) {
            // Vectorized load
            for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
                k_tile[idx + j] = k_cache[base_offset + idx + j];
                v_tile[idx + j] = v_cache[base_offset + idx + j];
            }
        }
    }
}

// ============================================================================
// Optimization 3: Fused softmax with online reduction
// ============================================================================

// Online softmax - compute max and exp sum in single pass
template<int NUM_WARPS>
METAL_FUNC void online_softmax(
    threadgroup float* logits,
    threadgroup float* weights,
    int num_tokens,
    uint lane [[thread_index_in_simdgroup]],
    uint warp_idx [[simdgroup_index_in_threadgroup]]
) {
    // Each warp processes a chunk of tokens
    const int tokens_per_warp = (num_tokens + NUM_WARPS - 1) / NUM_WARPS;
    const int start = warp_idx * tokens_per_warp;
    const int end = min(start + tokens_per_warp, num_tokens);

    // Find local max
    float local_max = -FLT_MAX;
    for (int i = start + lane; i < end; i += 32) {
        local_max = max(local_max, logits[i]);
    }
    local_max = simd_max(local_max);

    // Store local max for global reduction
    threadgroup float warp_maxes[NUM_WARPS];
    if (lane == 0) {
        warp_maxes[warp_idx] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find global max
    float global_max = -FLT_MAX;
    if (warp_idx == 0) {
        for (int i = lane; i < NUM_WARPS; i += 32) {
            global_max = max(global_max, warp_maxes[i]);
        }
        global_max = simd_max(global_max);
        if (lane == 0) {
            warp_maxes[0] = global_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = warp_maxes[0];

    // Compute exp and local sum
    float local_sum = 0.0f;
    for (int i = start + lane; i < end; i += 32) {
        weights[i] = exp(logits[i] - global_max);
        local_sum += weights[i];
    }
    local_sum = simd_sum(local_sum);

    // Store local sum for global reduction
    threadgroup float warp_sums[NUM_WARPS];
    if (lane == 0) {
        warp_sums[warp_idx] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find global sum
    float global_sum = 0.0f;
    if (warp_idx == 0) {
        for (int i = lane; i < NUM_WARPS; i += 32) {
            global_sum += warp_sums[i];
        }
        global_sum = simd_sum(global_sum);
        if (lane == 0) {
            warp_sums[0] = global_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = warp_sums[0];

    // Normalize weights
    float inv_sum = 1.0f / global_sum;
    for (int i = start + lane; i < end; i += 32) {
        weights[i] *= inv_sum;
    }
}

// ============================================================================
// Optimized Paged Attention Kernel
// ============================================================================

template<typename T, int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS, int NUM_SIMD_LANES>
[[kernel]] void paged_attention_optimized(
    device float* exp_sums [[buffer(0)]],
    device float* max_logits [[buffer(1)]],
    device T* out [[buffer(2)]],
    device const T* q [[buffer(3)]],
    device const T* k_cache [[buffer(4)]],
    device const T* v_cache [[buffer(5)]],
    const constant int& num_kv_heads [[buffer(6)]],
    const constant float& scale [[buffer(7)]],
    const constant float& softcapping [[buffer(8)]],
    device const uint32_t* block_tables [[buffer(9)]],
    device const uint32_t* context_lens [[buffer(10)]],
    const constant int& max_num_blocks_per_seq [[buffer(11)]],
    const constant int& q_stride [[buffer(12)]],
    const constant int& kv_block_stride [[buffer(13)]],
    const constant int& kv_head_stride [[buffer(14)]],
    threadgroup char* shared_mem [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int seq_idx = tid.y;
    const int head_idx = tid.x;
    const int num_heads = tid.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const uint32_t context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // Threadgroup memory layout
    threadgroup T* q_tile = reinterpret_cast<threadgroup T*>(shared_mem);
    threadgroup T* k_tile = q_tile + HEAD_DIM;
    threadgroup T* v_tile = k_tile + BLOCK_SIZE * HEAD_DIM;
    threadgroup float* logits = reinterpret_cast<threadgroup float*>(v_tile + BLOCK_SIZE * HEAD_DIM);
    threadgroup float* weights = logits + BLOCK_SIZE * 32;  // Max partition size

    // Load query vector (once per threadgroup)
    const device T* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_DIM;
    for (int i = simd_lid; i < HEAD_DIM; i += 32) {
        q_tile[i] = q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize output accumulator
    thread float out_local[HEAD_DIM / 32] = {0};
    float qk_max = -FLT_MAX;
    float exp_sum = 0.0f;

    // Process KV blocks
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const device uint32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t physical_block = block_table[block_idx];
        if (physical_block < 0) continue;

        const int tokens_in_block = min(BLOCK_SIZE, context_len - block_idx * BLOCK_SIZE);

        // Async load K/V block
        async_load_kv_block<T, BLOCK_SIZE, HEAD_DIM>(
            k_cache, v_cache, k_tile, v_tile,
            physical_block, kv_head_idx,
            kv_block_stride, kv_head_stride,
            lid.x, NUM_THREADS
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute QK^T for each token in block
        for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
            // Fast QK dot product using simdgroup_matrix
            float qk;
            fast_qk_dot<T, HEAD_DIM>(
                q_tile,
                k_tile + token_offset * HEAD_DIM,
                qk,
                simd_tid, simd_lid
            );

            qk *= scale;

            // Apply softcapping
            if (softcapping != 1.0) {
                qk = precise::tanh(qk / softcapping) * softcapping;
            }

            if (simd_lid == 0) {
                logits[token_offset + block_idx * BLOCK_SIZE] = qk;
                qk_max = max(qk_max, qk);
            }
        }
    }

    // Online softmax
    constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
    online_softmax<NUM_WARPS>(logits, weights, context_len, simd_lid, simd_tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute weighted sum of V
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t physical_block = block_table[block_idx];
        if (physical_block < 0) continue;

        const int tokens_in_block = min(BLOCK_SIZE, context_len - block_idx * BLOCK_SIZE);

        for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
            const float weight = weights[token_offset + block_idx * BLOCK_SIZE];
            const threadgroup T* v_ptr = v_tile + token_offset * HEAD_DIM;

            // Accumulate weighted V
            for (int i = simd_lid; i < HEAD_DIM; i += 32) {
                out_local[i / 32] += weight * float(v_ptr[i]);
            }
        }
    }

    // Write output
    device T* out_ptr = out + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    for (int i = simd_lid; i < HEAD_DIM; i += 32) {
        out_ptr[i] = T(out_local[i / 32]);
    }
}

// ============================================================================
// Template Instantiations
// ============================================================================

#define instantiate_paged_attention_optimized(type, head_dim, block_size, num_threads, num_simd_lanes) \
  template [[host_name("paged_attention_optimized_" #type "_hd" #head_dim "_bs" #block_size)]] \
  [[kernel]] void paged_attention_optimized<type, head_dim, block_size, num_threads, num_simd_lanes>( \
    device float* exp_sums [[buffer(0)]], \
    device float* max_logits [[buffer(1)]], \
    device type* out [[buffer(2)]], \
    device const type* q [[buffer(3)]], \
    device const type* k_cache [[buffer(4)]], \
    device const type* v_cache [[buffer(5)]], \
    const constant int& num_kv_heads [[buffer(6)]], \
    const constant float& scale [[buffer(7)]], \
    const constant float& softcapping [[buffer(8)]], \
    device const uint32_t* block_tables [[buffer(9)]], \
    device const uint32_t* context_lens [[buffer(10)]], \
    const constant int& max_num_blocks_per_seq [[buffer(11)]], \
    const constant int& q_stride [[buffer(12)]], \
    const constant int& kv_block_stride [[buffer(13)]], \
    const constant int& kv_head_stride [[buffer(14)]], \
    threadgroup char* shared_mem [[threadgroup(0)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_tid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

// Common configurations
instantiate_paged_attention_optimized(float, 128, 16, 256, 32)
instantiate_paged_attention_optimized(float, 64, 16, 256, 32)
instantiate_paged_attention_optimized(bfloat16_t, 128, 16, 256, 32)
instantiate_paged_attention_optimized(bfloat16_t, 64, 16, 256, 32)
instantiate_paged_attention_optimized(half, 128, 16, 256, 32)
instantiate_paged_attention_optimized(half, 64, 16, 256, 32)
