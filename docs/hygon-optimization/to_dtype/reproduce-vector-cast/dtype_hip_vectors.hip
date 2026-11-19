#include <hip/hip_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint32_t round_bf16(uint32_t bits) {
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return 0x7fc0u;
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

template <bool Widen>
__device__ __forceinline__ void cast_four(const void* source, void* destination,
                                         unsigned index) {
    if constexpr (Widen) {
        const uint2 packed = static_cast<const uint2*>(source)[index];
        static_cast<uint4*>(destination)[index] = make_uint4(
            packed.x << 16, packed.x & 0xffff0000u,
            packed.y << 16, packed.y & 0xffff0000u);
    } else {
        const uint4 words = static_cast<const uint4*>(source)[index];
        static_cast<uint2*>(destination)[index] = make_uint2(
            round_bf16(words.x) | (round_bf16(words.y) << 16),
            round_bf16(words.z) | (round_bf16(words.w) << 16));
    }
}

template <bool Widen, int Vectors>
__global__ void cast_vectors(const void* source, void* destination) {
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int vector = 0; vector < Vectors; ++vector) {
        const unsigned index = lane + vector * gridDim.x * blockDim.x;
        cast_four<Widen>(source, destination, index);
    }
}

template <bool Widen, int Vectors>
__global__ void cast_with_tail(const void* source, void* destination, unsigned count) {
    const unsigned first = blockIdx.x * blockDim.x * Vectors + threadIdx.x;
    #pragma unroll
    for (int vector = 0; vector < Vectors; ++vector) {
        const unsigned index = first + vector * blockDim.x;
        if (index < count / 4) {
            cast_four<Widen>(source, destination, index);
        } else if (index == count / 4) {
            for (unsigned i = index * 4; i < count; ++i) {
                if constexpr (Widen)
                    static_cast<uint32_t*>(destination)[i] =
                        uint32_t(static_cast<const uint16_t*>(source)[i]) << 16;
                else
                    static_cast<uint16_t*>(destination)[i] =
                        round_bf16(static_cast<const uint32_t*>(source)[i]);
            }
        }
    }
}

template <bool Widen>
void launch(const void* source, void* output, int threads, int vectors, hipStream_t stream) {
    const int blocks = 65536 / (threads * vectors);
    if (vectors == 1)
        hipLaunchKernelGGL((cast_vectors<Widen, 1>), blocks, threads, 0, stream, source, output);
    else if (vectors == 2)
        hipLaunchKernelGGL((cast_vectors<Widen, 2>), blocks, threads, 0, stream, source, output);
    else
        hipLaunchKernelGGL((cast_vectors<Widen, 4>), blocks, threads, 0, stream, source, output);
}

extern "C" void dtype_cast_262144(const void* source, void* output, int widen,
                                  int threads, int vectors, hipStream_t stream) {
    if (widen)
        launch<true>(source, output, threads, vectors, stream);
    else
        launch<false>(source, output, threads, vectors, stream);
}

template <bool Widen>
void launch_tail(const void* source, void* output, unsigned count,
                 int threads, int vectors, hipStream_t stream) {
    const unsigned width = threads * vectors * 4;
    const unsigned blocks = (count + width - 1) / width;
    if (vectors == 1)
        hipLaunchKernelGGL((cast_with_tail<Widen, 1>), blocks, threads, 0, stream, source, output, count);
    else if (vectors == 2)
        hipLaunchKernelGGL((cast_with_tail<Widen, 2>), blocks, threads, 0, stream, source, output, count);
    else
        hipLaunchKernelGGL((cast_with_tail<Widen, 4>), blocks, threads, 0, stream, source, output, count);
}

extern "C" void dtype_cast_dense(const void* source, void* output, unsigned count,
                                 int widen, int threads, int vectors, hipStream_t stream) {
    if (count == 262144)
        dtype_cast_262144(source, output, widen, threads, vectors, stream);
    else if (widen)
        launch_tail<true>(source, output, count, threads, vectors, stream);
    else
        launch_tail<false>(source, output, count, threads, vectors, stream);
}

template <int Vectors>
__global__ void cast_int64_float(const longlong2* source, float4* destination) {
    const unsigned first = blockIdx.x * blockDim.x * Vectors + threadIdx.x;
    #pragma unroll
    for (int vector = 0; vector < Vectors; ++vector) {
        const unsigned i = first + vector * blockDim.x;
        const longlong2 lo = source[2 * i];
        const longlong2 hi = source[2 * i + 1];
        destination[i] = make_float4(static_cast<float>(lo.x), static_cast<float>(lo.y),
                                     static_cast<float>(hi.x), static_cast<float>(hi.y));
    }
}

extern "C" void dtype_cast_int64_4194304(const void* source, void* output,
                                         int threads, int vectors, hipStream_t stream) {
    const auto* input = static_cast<const longlong2*>(source);
    auto* result = static_cast<float4*>(output);
    const int blocks = 1048576 / (threads * vectors);
    if (vectors == 1)
        hipLaunchKernelGGL((cast_int64_float<1>), blocks, threads, 0, stream, input, result);
    else if (vectors == 2)
        hipLaunchKernelGGL((cast_int64_float<2>), blocks, threads, 0, stream, input, result);
    else
        hipLaunchKernelGGL((cast_int64_float<4>), blocks, threads, 0, stream, input, result);
}
