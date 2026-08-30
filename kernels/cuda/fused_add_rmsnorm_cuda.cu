#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace {

constexpr int kHiddenSize = 1536;
constexpr int kThreadsPerBlock = 256;
constexpr int kItemsPerThread = kHiddenSize / kThreadsPerBlock;

static_assert(
    kHiddenSize % kThreadsPerBlock == 0,
    "hidden size must be divisible by threads per block"
);


__global__ void fused_add_rmsnorm_bf16_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int64_t rows,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    /*
     * One CUDA block handles one token row:
     *
     *   256 threads/block × 6 values/thread = 1536 hidden values.
     *
     * Each thread keeps its six (x + residual) values in registers while
     * the block performs the RMS reduction. This avoids materializing the
     * residual-add result in global memory.
     */
    float values[kItemsPerThread];
    float square_sum = 0.0f;

    const int64_t row_offset =
        static_cast<int64_t>(row) * kHiddenSize;

    #pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int col =
            tid + item * kThreadsPerBlock;

        const int64_t index =
            row_offset + col;

        const float x_value =
            __bfloat162float(x[index]);

        const float residual_value =
            __bfloat162float(residual[index]);

        const float summed =
            x_value + residual_value;

        values[item] = summed;
        square_sum += summed * summed;
    }

    /*
     * Shared-memory reduction:
     * 256 partial sums -> one sum for the token row.
     *
     * This is deliberately straightforward CUDA rather than an aggressively
     * hand-tuned warp-shuffle implementation. Phase 5.5 profiling will tell
     * us where this native implementation loses time relative to Triton.
     */
    __shared__ float shared[kThreadsPerBlock];

    shared[tid] = square_sum;
    __syncthreads();

    for (
        int stride = kThreadsPerBlock / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        const float mean_square =
            shared[0]
            / static_cast<float>(kHiddenSize);

        shared[0] =
            rsqrtf(mean_square + eps);
    }

    __syncthreads();

    const float inv_rms = shared[0];

    #pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int col =
            tid + item * kThreadsPerBlock;

        const int64_t index =
            row_offset + col;

        const float weight_value =
            __bfloat162float(weight[col]);

        const float result =
            values[item]
            * inv_rms
            * weight_value;

        output[index] =
            __float2bfloat16_rn(result);
    }
}


void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight
) {
    TORCH_CHECK(
        x.is_cuda(),
        "x must be a CUDA tensor"
    );

    TORCH_CHECK(
        residual.is_cuda(),
        "residual must be a CUDA tensor"
    );

    TORCH_CHECK(
        weight.is_cuda(),
        "weight must be a CUDA tensor"
    );

    TORCH_CHECK(
        x.scalar_type() == torch::kBFloat16,
        "x must use torch.bfloat16"
    );

    TORCH_CHECK(
        residual.scalar_type() == torch::kBFloat16,
        "residual must use torch.bfloat16"
    );

    TORCH_CHECK(
        weight.scalar_type() == torch::kBFloat16,
        "weight must use torch.bfloat16"
    );

    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous"
    );

    TORCH_CHECK(
        residual.is_contiguous(),
        "residual must be contiguous"
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous"
    );

    TORCH_CHECK(
        x.dim() == 2,
        "x must have shape [tokens, hidden_size]"
    );

    TORCH_CHECK(
        residual.sizes() == x.sizes(),
        "residual must have the same shape as x"
    );

    TORCH_CHECK(
        x.size(1) == kHiddenSize,
        "this Phase 5.4 kernel is specialized for hidden_size=1536"
    );

    TORCH_CHECK(
        weight.dim() == 1
        && weight.size(0) == kHiddenSize,
        "weight must have shape [1536]"
    );
}

}  // namespace


torch::Tensor fused_add_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps
) {
    check_inputs(
        x,
        residual,
        weight
    );

    const int64_t rows =
        x.size(0);

    auto output =
        torch::empty_like(x);

    const dim3 grid(
        static_cast<unsigned int>(rows)
    );

    const dim3 block(
        kThreadsPerBlock
    );

    fused_add_rmsnorm_bf16_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()
        ),
        reinterpret_cast<const __nv_bfloat16*>(
            residual.data_ptr<at::BFloat16>()
        ),
        reinterpret_cast<const __nv_bfloat16*>(
            weight.data_ptr<at::BFloat16>()
        ),
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()
        ),
        rows,
        static_cast<float>(eps)
    );

    const cudaError_t error =
        cudaGetLastError();

    TORCH_CHECK(
        error == cudaSuccess,
        "CUDA kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return output;
}


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
) {
    m.def(
        "fused_add_rmsnorm",
        &fused_add_rmsnorm_cuda,
        "Fused residual add + RMSNorm (CUDA, BF16)"
    );
}
