#include "gs_cuda.cuh"

__device__ __inline__ void direct_substitution(half2* data, const float target_amp) {
    // Directly substitute amplitude
    float re = __low2float(*data);
    float im = __high2float(*data);
    float factor = target_amp*rsqrtf(fmaf(re , re , im * im));
    *data =  __floats2half2_rn(factor*re, factor*im);
}

__device__ __inline__ void selected_substitution(half2* data, const float target_amp, const float threshold) {
    // Only substitute if the intensity is greater than a threshold, allowing background evolve freely
    // Use to resolve ghost traps
    // https://doi.org/10.1364/OL.36.003657
    float re = __low2float(*data);
    float im = __high2float(*data);
    float inten_r = fmaf(re , re , im * im);
    float sel = (inten_r > threshold) ? 1.0f : 0.0f;
    float factor = fmaf(sel,fmaf(target_amp,rsqrtf(inten_r),-1.f),1.f);
    *data = __floats2half2_rn(re*factor, im*factor);
}

__device__ __inline__ void weighted_substitution(half2* data, const float target_amp, float* previous_amp) {
    // Weighted substitution to enhance convergence rate
    // Check "OneNote/Setting up Hardware Vol. 3/Spatial Light Modulator (SLM)/ Oliver's Note": https://tinyurl.com/4ftt92m6
    float re = __low2float(*data);
    float im = __high2float(*data);
    float inten_r = fmaf(re , re , im * im);
    float weighted_intensity = target_amp*rsqrtf(fmaf(-0.8f,inten_r/fmaf(*previous_amp,*previous_amp,1e-10f),1.0f));
    *previous_amp = weighted_intensity;
    weighted_intensity *= rsqrtf(inten_r);
    *data = __floats2half2_rn(weighted_intensity*re, weighted_intensity*im);
}

__device__ __inline__ void selected_weighted_substitution(half2* data, const float target_amp, float* previous_amp, const float threshold) {
    // Combination of weighted substitution and selected substitution
    float re = __low2float(*data);
    float im = __high2float(*data);
    float inten_r = fmaf(re , re , im * im);
    float sel = (inten_r > threshold) ? 1.0f : 0.0f;
    float weighted_intensity = target_amp*rsqrtf(fmaf(-0.8f,inten_r/fmaf(*previous_amp,*previous_amp,1e-10f),1.0f));
    *previous_amp = weighted_intensity;
    weighted_intensity = fmaf(sel, fmaf(weighted_intensity,rsqrtf(inten_r),-1.f), 1.f);
    *data = __floats2half2_rn(weighted_intensity*re, weighted_intensity*im);
}

__global__ void amp_substitution(half2* data, half*  amp_list, const bool if_target, half* oamp_list, const float thres) {
    // The substitute the amplitude after FFT in GS iteration
    // The kernel processes two pixels at a time to optimize memory access pattern

    // data: the data after FFT
    // amp_list: the (new if weighted) target amplitudes
    // if_target: ture for the target plane substitution
    // oamp_list: the list of original target amplitudes; used only when there is a weighted substitution
    // thres: the threshold value for selected substitution only
    // type: index for the substitution method; 0: direct, 1: selected, 2: weighted, 3: selected weighted

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // struct half4 {half2 data1; half2 data2;} ;
    half4 data4;
    *reinterpret_cast<uint2*>(&data4) = reinterpret_cast<uint2*>(data)[idx];

    half2 amp2 = reinterpret_cast<half2*>(amp_list)[idx];
    float scale = __low2float(amp2);
    float scale1 = __high2float(amp2);

    if(if_target){
        if constexpr (GS_MODE==GS_MODE_DIRECT) {
            // Direct substitution
            direct_substitution(&data4.data1,scale);
            direct_substitution(&data4.data2,scale1);
        }else if constexpr (GS_MODE==GS_MODE_SELECTED) {
            // Selected substitution
            selected_substitution(&data4.data1,scale,thres);
            selected_substitution(&data4.data2,scale1,thres);
        }else if constexpr (GS_MODE==GS_MODE_WEIGHTED) {
            // Weighted substitution
            half2 oamp2 = reinterpret_cast<half2*>(oamp_list)[idx];
            float oamp = __low2float(oamp2);
            float oamp1 = __high2float(oamp2);
            weighted_substitution(&data4.data1,oamp,&scale);
            weighted_substitution(&data4.data2,oamp1,&scale1);
            reinterpret_cast<half2*>(amp_list)[idx] = __floats2half2_rn(scale,scale1);
        }else if constexpr (GS_MODE==GS_MODE_SELECTED_WEIGHTED) {
            // Selected-weighted substitution
            half2 oamp2 = reinterpret_cast<half2*>(oamp_list)[idx];
            float oamp = __low2float(oamp2);
            float oamp1 = __high2float(oamp2);
            weighted_substitution(&data4.data1,oamp,&scale);
            weighted_substitution(&data4.data2,oamp1,&scale1);
            reinterpret_cast<half2*>(amp_list)[idx] = __floats2half2_rn(scale,scale1);
        }
    }else{
        direct_substitution(&data4.data1,scale);
        direct_substitution(&data4.data2,scale1);
    }
    reinterpret_cast<uint2*>(data)[idx] = *reinterpret_cast<uint2*>(&data4);
}


__device__ __inline__ unsigned char phase_to_uchar(const float phase){
    // convert phase from radians to SLM levels
    return static_cast<unsigned char>(fmaxf(nearbyintf(fmaf(phase,M_1_PIf , 1.f)*128.0f), 0.0f));
}

__global__ void get_phase(unsigned char* __restrict__ sout_phase_cuda, const half2 * __restrict__ s_data){
    // convert phase from radians to SLM levels in vectors of 4
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float4 s_data4 = reinterpret_cast<const float4*>(s_data)[idx];
    const half2 slm_data  = *reinterpret_cast<const half2*>(&s_data4.x);
    const half2 slm_data1 = *reinterpret_cast<const half2*>(&s_data4.y);
    const half2 slm_data2 = *reinterpret_cast<const half2*>(&s_data4.z);
    const half2 slm_data3 = *reinterpret_cast<const half2*>(&s_data4.w);
    reinterpret_cast<uchar4*>(sout_phase_cuda)[idx]= make_uchar4(
        phase_to_uchar(atan2f(__high2float(slm_data), __low2float(slm_data))),
        phase_to_uchar(atan2f(__high2float(slm_data1), __low2float(slm_data1))),
        phase_to_uchar(atan2f(__high2float(slm_data2), __low2float(slm_data2))),
        phase_to_uchar(atan2f(__high2float(slm_data3), __low2float(slm_data3)))
    );
}


__global__ void float2halfM(const float* __restrict__ in, half* __restrict__ out, const float scale) {
    // load 2 floats, multiply by scale, store to a half array
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float2 indata2 = reinterpret_cast<const float2*>(in)[idx];
    reinterpret_cast<half2*>(out)[idx] = __floats2half2_rn(indata2.x * scale, indata2.y * scale);
}


__global__ void generate_initial_target(const half* __restrict__ amp_host, half* __restrict__ target_amp, half2* __restrict__ target_complex,
                        const float* __restrict__ random_phase_cuda, const float scale) {
    // Load the target field data with preset amplitudes and random phases
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float2 rand_phase2 = reinterpret_cast<const float2*>(random_phase_cuda)[idx];
    const half2 amp_host2 = reinterpret_cast<const half2*>(amp_host)[idx];
    const float ampx = __low2float(amp_host2) * scale;
    const float ampy = __high2float(amp_host2) * scale;
    float sinx, cosx;
    float siny, cosy;
    sincosf(rand_phase2.x, &sinx, &cosx);
    sincosf(rand_phase2.y, &siny, &cosy);
    reinterpret_cast<half2*>(target_amp)[idx] = __floats2half2_rn(ampx, ampy);
    reinterpret_cast<half4*>(target_complex)[idx] = {
        __floats2half2_rn(ampx * cosx,ampx * sinx),
        __floats2half2_rn(ampy * cosy,ampy * siny)
    };
}

