#ifndef _gs_cuda_included_
#define _gs_cuda_included_

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <cufftXt.h>
#include "cufft_utils.h"
#include "CONFIGURATIONS.h"

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>


#define M_1_PIf 0.31830988618f
#define M_PIf 3.14159274101257f

struct __align__(8) half4 {
    __half2 data1;  // values 0, 1
    __half2 data2;  // values 2, 3
};


extern __global__ void amp_substitution(half2* data, half*  amp_list, const bool if_target, half* oamp_list, const float thres);
extern __global__ void get_phase(unsigned char* __restrict__ sout_phase_cuda, const half2 * __restrict__ s_data);
extern __global__ void float2halfM(const float* __restrict__ in, half* __restrict__ out, const float scale);
extern __global__ void generate_initial_target(const half* __restrict__ amp_host, half* __restrict__ target_amp, half2* __restrict__ target_complex,
                        const float* __restrict__ random_phase_cuda, const float scale);

__device__ __forceinline__ __half intensity_half2(__half2 x){
    // Compute the intensity of a complex number
    const __half2 x2 = __hmul2(x, x);
    return __low2half(__hadd2(x2, __lowhigh2highlow(x2)));
}

struct IntensitySumOp {
    // Operator for reduce sum of field intensities
    const half2* __restrict__ data;

    __device__ __forceinline__
    float operator()(int idx) const
    {
        // One output index consumes two complex half2 values.
        const half2 z0 = data[2 * idx];
        const half2 z1 = data[2 * idx + 1];
        // Sum both lanes into the FP32 reduction accumulator.
        return __half2float(intensity_half2(z0)) +
               __half2float(intensity_half2(z1));
    }
};            

struct IntensitySquareSumOp {
    // Operator for reduce sum of field intensities
    const half2* __restrict__ data;

    __device__ __forceinline__
    float operator()(int idx) const
    {
        // One output index consumes two complex half2 values.
        const half2 z0 = data[2 * idx];
        const half2 z1 = data[2 * idx + 1];
        float int1 = __half2float(intensity_half2(z0));
        float int2 = __half2float(intensity_half2(z1));
        // Sum both lanes into the FP32 reduction accumulator.
        return int1*int1 + int2*int2;
    }
};         

struct SquareSumOp {
    // Operator for reduce sum of field intensities
    const half* __restrict__ data;

    __device__ __forceinline__
    float operator()(int idx) const
    {
        // One output index consumes two complex half2 values.
        const float z0 = __half2float(data[2 * idx]);
        const float z1 = __half2float(data[2 * idx + 1]);
        // Sum both lanes into the FP32 reduction accumulator.
        return z0*z0*z0*z0 + z1*z1*z1*z1;
    }
};     

struct IntensityMultiplySumOp {
    // Operator for reduce sum of fidelity, calculated by element-wise H*T,
    // where H and T are intensities matrices of the reconstructed and target holograms

    const half2* __restrict__ data;
    const half*  __restrict__ amp_list;

    __device__ __forceinline__
    float operator()(int idx) const
    {
        // One output index consumes two complex half2 values.
        const half2 z0 = data[2 * idx];
        const half2 z1 = data[2 * idx + 1];

        // amp_list[2*idx] and amp_list[2*idx + 1].
        const half2 amp =
            reinterpret_cast<const half2*>(amp_list)[idx];

        // Per-lane amplitude squared.
        const half2 amp_intensity =
            __hmul2(amp, amp);

        // intensity_half2(z) = real(z)^2 + imag(z)^2.
        const half2 reconstructed_intensity =
            __halves2half2(
                intensity_half2(z0),
                intensity_half2(z1)
            );

        // Two weighted intensity values, held in the two lanes.
        const half2 result =
            __hmul2(
                amp_intensity,
                reconstructed_intensity
            );

        // Sum both lanes into the FP32 reduction accumulator.
        return __half2float(__low2half(result)) +
               __half2float(__high2half(result));
    }
};


#endif