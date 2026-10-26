#define _USE_MATH_DEFINES
#include "gs_cuda.cuh"
#include "server.h"
#include "image_process.hpp"
#include "Blink_C_wrapper.h"
#include <conio.h>
#include <cmath>
#include <array>
#include <random>
#include <complex>
#include <iostream>
#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>
#include "cufft_utils.h"
#include <math.h>
#include <fstream>
#include <windows.h>
// #include <opencv2/opencv.hpp>
#include <chrono>
#include <cuda_fp16.h>
#include <cufftXt.h>
#include "device_launch_parameters.h"
#include "CONFIGURATIONS.h"


#include <thread>
#include <cstring>
#include <fstream>



cufftHandle fftplan = 0;
cudaError_t eCudaErr = cudaSuccess;

constexpr float focal_length = FOCAL_LENGTH;
constexpr char WFC_file_path_532[] = WFC_FILE_PATH_532;
constexpr char WFC_file_path_840[] = WFC_FILE_PATH_840;
constexpr int iteration_num = MAX_ITERATION;
constexpr int TriggerTimeout = TRIGGER_TIMEOUT;
unsigned char* Blank = new unsigned char[total_pixel];


float random_phase[total_pixel];
float * fout_amp_host;
float * random_phase_cuda;
unsigned char *sout_phase_host;

std::thread serverThread;
// Construct a Blink_SDK instance
unsigned int n_boards_found = 0U;
int constructed_okay = 0;

half* half_host;
float* float_host;
float* fout_amp_cuda;
float* intermediate_host;
float* intermediate_amp_host;
cufftComplex* intermediate2_host;
half2* s_data = nullptr;
half2* f_data = nullptr;
float beam_normalizer = 0.f;
float fidelity = 0.f;

//using namespace cv;

inline float gaussian(const float x, const float mu, const float sigma) {
    // unit-area gaussian
    return exp(-(x-mu)*(x-mu)/(sigma*sigma))/sqrtf(M_PIf*sigma*sigma);
}

inline float gaussian_beam(const float x, const float y,
        const float mux, const float muy, const float sigmax, const float sigmay) {
    // 2D gaussian
    return gaussian(x,mux,sigmax)*gaussian(y,muy,sigmay);
}

static void clean_exit() {
    CUFFT_CALL(cufftDestroy(fftplan));
    CUDA_RT_CALL(cudaDeviceReset());
    Delete_SDK(); // SLM related code
    printf("Main thread exits successfully\n");
    serverThread.join();
}

static void dummy_array_generator() {
    for (int i=0; i<DUMMY_SIDE_532;i++){
        for (int j=0;j<DUMMY_SIDE_532;j++){
            holo_params_532.mux[i*DUMMY_SIDE_532+j] = OFFSET_X_532+DUMMY_PITCH_532*i;
            holo_params_532.muy[i*DUMMY_SIDE_532+j] = OFFSET_Y_532+DUMMY_PITCH_532*j;
        }
    }
    holo_params_532.ptnum = DUMMY_SIDE_532*DUMMY_SIDE_532;
    holo_params_532.flag = FLAG_NEW;

    for (int i=0; i<DUMMY_SIDE_840;i++){
        for (int j=0;j<DUMMY_SIDE_840;j++){
            holo_params_840.mux[i*DUMMY_SIDE_840+j] = OFFSET_X_840+DUMMY_PITCH_840*i;
            holo_params_840.muy[i*DUMMY_SIDE_840+j] = OFFSET_Y_840+DUMMY_PITCH_840*j;
        }
    }
    holo_params_840.ptnum = DUMMY_SIDE_840*DUMMY_SIDE_840;
    holo_params_840.flag = FLAG_NEW;
}

static void random_phase_generator(float * random_phase) {
    for (int i = 0; i < total_pixel; i++) {
        random_phase[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX)*M_PIf*2.f;
    }
}


static int fidelity_checker(half2* f_data, half* target_amp_cuda, const float threshold, float* fidelity) {
    // Calculate the fidelity of the reconstructed hologram H compared to the target T (both in intensities);
    // Computed by summing the element-wise H*T after normalizing H by its total intensity; T should be normalized in advance.

    const int n_outputs = total_pixel/2; //  Number of threads, divided by 2 due to half2 type
    // iterators for reduction sum
    auto first = thrust::make_counting_iterator<int>(0); 
    auto last  = first + n_outputs;

    IntensityMultiplySumOp op{
        // reduction sum with element-wise H*T
        f_data,
        target_amp_cuda
    };

    IntensitySumOp op2{
        // reduction sum with intensity
        f_data
    };

    IntensitySquareSumOp op3{
        // reduction sum with intensity squared
        f_data
    };

    SquareSumOp op4{
        // reduction sum with intensity squared
        target_amp_cuda
    };

    const float f_intensity_normalizer =
        1.f/thrust::transform_reduce(
            thrust::device,
            first,
            last,
            op2,
            0.0f,
            thrust::plus<float>()
        );

    const float mutiply_sum =
        thrust::transform_reduce(
            thrust::device,
            first,
            last,
            op,
            0.0f,
            thrust::plus<float>()
        )*2.f*f_intensity_normalizer;

    const float denom = thrust::transform_reduce(
            thrust::device,
            first,
            last,
            op3,
            0.0f,
            thrust::plus<float>()
        )*f_intensity_normalizer*f_intensity_normalizer+
        thrust::transform_reduce(
            thrust::device,
            first,
            last,
            op4,
            0.0f,
            thrust::plus<float>()
        );

    *fidelity = mutiply_sum/denom;
    if (*fidelity > threshold) {
        // If fidelity is greater than threshold, return 1
        return 1;
    }
    return 0;
}

static int GS_kernel(half* beam_amp_cuda, half* target_amp_cuda, const bool is_last, half* omap_list){
    // perform a single GS iteration
    // beam_amp_cuda: incident beam amplitude array
    // target_amp_cuda: target amplitude mask (in last iteration)
    // is_last: whether this is the last iteration
    // omap_list: original target amplitude mask


    cufftXtExec(fftplan, f_data, s_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    amp_substitution<<<total_pixel/256/2,256>>>(s_data, beam_amp_cuda,false,
        omap_list,0.03f*sqrtf(total_pixel));
    cudaDeviceSynchronize();
    cufftXtExec(fftplan, s_data, f_data, CUFFT_INVERSE);
    cudaDeviceSynchronize();


    if (!is_last){
        if constexpr (IF_GS_THRESHOLD){
            // break when the fidelity threshold is reached
            if (fidelity_checker(f_data, omap_list, GS_THRESHOLD, &fidelity)) return 1;
        }
        amp_substitution<<<total_pixel/256/2,256>>>(f_data, target_amp_cuda,true,
            omap_list,0.03f*sqrtf(total_pixel));
        cudaDeviceSynchronize();
    }
    return 0;
}

void GS_execution(const int iteration_count, half* beam_amp_cuda, half* target_amp_cuda, half* omap_list){
    // perform multiple GS iterations

    // auto ts = std::chrono::steady_clock::now();
    for (int i = 0; i < iteration_count-1; i++) {
        // a single GS iteration
        // break when the fidelity threshold is reached
        if(GS_kernel(beam_amp_cuda, target_amp_cuda, i == iteration_count-1  ,omap_list)) break;
    }
    cudaDeviceSynchronize();
    printf("Fidality of the reconstructed hologram: %f\n", fidelity);
    // auto tce = std::chrono::steady_clock::now();
    // extract phase
    get_phase<<<total_pixel/1024,256>>>(sout_phase_host,s_data);
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

void defocus_mask(unsigned char *sout_phase_host, float focal_length, float wavelength, float defocus){
    // add defocus phase mask to SLM phase array
    float factor = 0.5f*defocus/wavelength/focal_length/focal_length;
    for (int i=0; i < 1024; i++){
        float x = (i-511.5f)*17e-3f;
        for (int j = 0; j< 1024; j++){
            float y = (j-511.5f)*17e-3f;
            float phase = factor * (x*x+y*y);
            phase = phase - (int)phase+ 1.f;
            sout_phase_host[i*1024+j] += static_cast<unsigned char>(round(phase*256.0f));
        }
    }
}

inline void fftshift_load2half(const int x, const int y, const float value, half* array) {
    // perform FFTshift and store value into half-typed array
    const int sx = (x + nx/2) & (nx - 1);
    const int sy = (y + ny/2) & (ny - 1);
    array[sy*nx + sx] = __float2half_rn(value);
}

void generate_incident_ampmask(const float beam_radius, half* output){
    // generate incident beam amplitude mask based on beam radius
    beam_normalizer = 0.f;
    for (int i = 0; i < total_pixel; i++) {
        const int y = i / nx;
        const int x = i % nx;
        const float bamp = gaussian_beam(static_cast<float>(x), static_cast<float>(y),
                511.5,511.5,beam_radius,beam_radius);
        float_host[i] = bamp;
        beam_normalizer += bamp * bamp;
    }
    beam_normalizer = 1.f / std::sqrt(beam_normalizer);
    float2halfM<<<total_pixel/256/2,256>>>(float_host, output,beam_normalizer);
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

static void save_hologram(const int &channel, const char* filename){
    // save hologram to file
    FILE* fp = fopen(filename, "wb");
    fwrite(&channel, sizeof(int), 1, fp);
    fwrite(sout_phase_host, 1, total_pixel, fp);
    fclose(fp);
}

static void load_hologram(int &channel, const char* filename){
    // save hologram to file
    FILE* fp = fopen(filename, "rb");
    fread(&channel, sizeof(int), 1, fp);
    fread(sout_phase_host, 1, total_pixel, fp);
    fclose(fp);
}


static void generate_hologram(hologram_params& params){
    if (params.flag == FLAG_NEW) {
        // Generate hologram for 532 SLM
        float amp_normalizer = 0.f;
        std::memset(half_host,0,sizeof(half_host));
        for (int i = 0; i<params.ptnum;i++){
            fftshift_load2half(params.mux[i],params.muy[i],params.amp[i],half_host); // load target amplitude in a fftshifted array
            // TODO: Here we assume the tweezers have zero waist at atoms' plane; need to be changed later
            amp_normalizer += params.amp[i]*params.amp[i];
        }
        amp_normalizer = 1.f / std::sqrt(amp_normalizer);
        generate_initial_target<<<total_pixel/2/256,256>>>(half_host, params.target_amp_cuda,f_data, random_phase_cuda,amp_normalizer);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        cudaMemcpy(params.oamp_cuda,params.target_amp_cuda,total_pixel*sizeof(half),cudaMemcpyDeviceToDevice);
        GS_execution(iteration_num,params.beam_amp_cuda,params.target_amp_cuda,params.oamp_cuda);
        // Write into SLM
        for (int i = 0; i < total_pixel; i++){
            sout_phase_host[i] = sout_phase_host[i]+params.WFCimage[i];
        }
        if (params.is_defocus) defocus_mask(sout_phase_host, focal_length, params.wavelength, params.defocus);
        if (EXTERNAL_TRIGGER && machine_state == MACHINE_DUMMY) SetWaitForTrigger(params.board_index, 0);
        Write_image(params.board_index, sout_phase_host, TriggerTimeout);
        ImageWriteComplete(params.board_index, TriggerTimeout);
        if (EXTERNAL_TRIGGER && machine_state == MACHINE_DUMMY) SetWaitForTrigger(params.board_index, 1);
        params.flag = FLAG_NULL;
    }
}



bool initialize() {
    // Initialization
    Create_SDK(&n_boards_found, &constructed_okay); // Acquire SLMs handles
    if (constructed_okay != 1){
        // return of 1 means okay
        std::cout << "SLM Initialization Failed: "<<constructed_okay << '\n';
        printf("\nError: %s\n",Get_last_error_message());
        CUFFT_CALL(cufftDestroy(fftplan));
        CUDA_RT_CALL(cudaDeviceReset());
        return true;
    }
    printf("Number of Boards detected: %d\n", n_boards_found); // Board 1 is the 532 nm SLM; Board 2 is the 840 nm SLM
    if constexpr (SLM_type == SLMType::SLM_532){
        if (n_boards_found != 1) {
            std::cout << "Wrong SLM type: Only 532 SLM expected but " << n_boards_found << " boards detected"<< '\n';
            CUFFT_CALL(cufftDestroy(fftplan));
            CUDA_RT_CALL(cudaDeviceReset());
            return true;
        }else{
            holo_params_532.board_index = 1;
        }
    }
    if constexpr (SLM_type == SLMType::SLM_840){
        if (n_boards_found != 1) {
            std::cout << "Wrong SLM type: Only 840 SLM expected but " << n_boards_found << " boards detected"<< '\n';
            CUFFT_CALL(cufftDestroy(fftplan));
            CUDA_RT_CALL(cudaDeviceReset());
            return true;
        }else{
            holo_params_840.board_index = 1;
        }
    }
    if constexpr (SLM_type == SLMType::SLM_both){
        if (n_boards_found != 2) {
            std::cout << "Wrong SLM type: Both SLMs expected but " << n_boards_found << " boards detected"<< '\n';
            CUFFT_CALL(cufftDestroy(fftplan));
            CUDA_RT_CALL(cudaDeviceReset());
            return true;
        }else{
            holo_params_532.board_index = 1;
            holo_params_840.board_index = 2;
        }
    }

    // Set SLM configurations
    for (int i = 1; i < n_boards_found + 1; i++) {
        // board_number starts from 1
        SetWaitForTrigger(i, EXTERNAL_TRIGGER);
        SetFlipImmediate(i, FLIP_IMMEDIATE);
        SetOutputPulse(i, OUTPUT_PULSE);
    }

    // Load Wavefront Correction files
    std::ifstream WFCfile_532(WFC_file_path_532, std::ios::in | std::ios::binary);
    WFCfile_532.read(reinterpret_cast<char*>(holo_params_532.WFCimage), total_pixel);
    WFCfile_532.close();
    std::ifstream WFCfile_840(WFC_file_path_840, std::ios::in | std::ios::binary);
    WFCfile_840.read(reinterpret_cast<char*>(holo_params_840.WFCimage), total_pixel);
    WFCfile_840.close();
    std::thread serverThread(TCP_server);

    // Allocate GPU memory
    memset(Blank, 0, total_pixel);

    if constexpr (SLM_type != SLMType::SLM_840) {
        CUDA_RT_CALL(cudaMalloc((void**)&holo_params_532.beam_amp_cuda, sizeof(half) * total_pixel));
        CUDA_RT_CALL(cudaMalloc((void**)&holo_params_532.target_amp_cuda, sizeof(half) * total_pixel));
        CUDA_RT_CALL(cudaMalloc((void**)&holo_params_532.oamp_cuda, sizeof(half) * total_pixel));
    }
    if constexpr (SLM_type != SLMType::SLM_532) {
        CUDA_RT_CALL(cudaMalloc((void**)&holo_params_840.beam_amp_cuda, sizeof(half) * total_pixel));
        CUDA_RT_CALL(cudaMalloc((void**)&holo_params_840.target_amp_cuda, sizeof(half) * total_pixel));
        CUDA_RT_CALL(cudaMalloc((void**)&holo_params_840.oamp_cuda, sizeof(half) * total_pixel));
    }
    CUDA_RT_CALL(cudaMallocHost((void**)&fout_amp_host, sizeof(float) * total_pixel));
    CUDA_RT_CALL(cudaMallocHost((void **)&half_host, sizeof(half) * total_pixel));
    CUDA_RT_CALL(cudaMallocHost((void **)&float_host, sizeof(float) * total_pixel));
    CUDA_RT_CALL(cudaMallocHost((void **)&intermediate_host, sizeof(float) * total_pixel));
    CUDA_RT_CALL(cudaMallocHost((void **)&intermediate_amp_host, sizeof(float) * nx));
    CUDA_RT_CALL(cudaMallocHost((void **)&intermediate2_host, sizeof(cufftComplex) * total_pixel));
    CUDA_RT_CALL(cudaMallocHost((void **)&sout_phase_host, sizeof(char) *total_pixel));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&random_phase_cuda), total_pixel * sizeof(float)));
    random_phase_generator(random_phase);
    CUDA_RT_CALL(cudaMemcpy(random_phase_cuda, random_phase, total_pixel * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuFFT plan and containers
    CUFFT_CALL(cufftCreate(&fftplan));
    size_t ws = sizeof(half2) * total_pixel;
    long long sides[2]={nx,ny};
    CUFFT_CALL(cufftXtMakePlanMany(fftplan, 2, sides, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&s_data), sizeof(half2) * total_pixel));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&f_data), sizeof(half2) * total_pixel));

    // Generate incident beam amplitude mask
    if constexpr (SLM_type != SLMType::SLM_840) generate_incident_ampmask(holo_params_532.beam_radius, holo_params_532.beam_amp_cuda);
    if constexpr (SLM_type != SLMType::SLM_532) generate_incident_ampmask(holo_params_840.beam_radius, holo_params_840.beam_amp_cuda);
    loadLinearMatrix();
    return false; // return of false means okay
}


// --------------------------------------------------------------------------------------------------------
// Main
int main(int argc, char *argv[]) {
    // srand( time(NULL) );
    if (initialize()){
        std::cout << "Initialization Failed\n";
        return 1;
    }

    while (machine_state != MACHINE_TERMINATED){
        switch (machine_state){
            case MACHINE_IDLE:
                // wait for server commands
                if (_kbhit()) {
                    // terminate the program when ESC is pressed
                    char ch = _getch();
                    if (ch == 27) { // ESC
                        machine_state = MACHINE_TERMINATING;
                    }
                }
                break;

            case MACHINE_TERMINATING:
                clean_exit();
                machine_state = MACHINE_TERMINATED;
                break;
            
            case MACHINE_ZERO:
                holo_params_532.ptnum = 0;
                holo_params_840.ptnum = 0;
                for (int i = 0; i < n_boards_found; i++) {
	                if constexpr (EXTERNAL_TRIGGER) SetWaitForTrigger(i, 0);
                    Write_image(i, Blank, TriggerTimeout);
                    ImageWriteComplete(i, TriggerTimeout);
	                if constexpr (EXTERNAL_TRIGGER) SetWaitForTrigger(i, 1);
                }
                machine_state = MACHINE_IDLE;
                break;
         
            case MACHINE_POSITION_CALIBRATING:
                if (holo_params_532.ptnum) positionCalibration(image_buffer, holo_params_532);
                else positionCalibration(image_buffer, holo_params_840);
                machine_state = MACHINE_IDLE;
                break;

            case MACHINE_POSITION_CALIBRATING_ARMED:
                // server waiting for incoming iamge
                break;
                
            case MACHINE_INTENSITY_CALIBRATING:
                if (holo_params_532.ptnum) intensityCalibration(image_buffer, holo_params_532);
                else intensityCalibration(image_buffer, holo_params_840);
                machine_state = MACHINE_IDLE;
                break;

            case MACHINE_INTENSITY_CALIBRATING_ARMED:
                // server waiting for incoming iamge
                break;

            case MACHINE_DUMMY:
                dummy_array_generator();
                machine_state = MACHINE_GENERATING;
                break;

            case MACHINE_GENERATING:
                if constexpr (SLM_type != SLMType::SLM_840) generate_hologram(holo_params_532);
                if constexpr (SLM_type != SLMType::SLM_532) generate_hologram(holo_params_840);
                machine_state = MACHINE_IDLE;
                break;

            case MACHINE_SAVE_HOLOGRAM:
                if (holo_params_532.ptnum) save_hologram(532, phase_mask_name);
                else save_hologram(840, phase_mask_name);
                machine_state = MACHINE_IDLE;
                break;

            case MACHINE_LOAD_HOLOGRAM:
                int channel, which_board;
                load_hologram(channel, phase_mask_name);
                if (channel == 532) {
                    which_board = holo_params_532.board_index;
                }else if (channel == 840) {
                    which_board = holo_params_840.board_index;
                }else {
                    std::cerr << "Unknown channel: "<<channel<<" during hologram loading;" << std::endl;
                    machine_state = MACHINE_IDLE;
                    break;
                }
                if constexpr (EXTERNAL_TRIGGER) SetWaitForTrigger(which_board, 0);
                Write_image(which_board, sout_phase_host, TriggerTimeout);
                ImageWriteComplete(which_board, TriggerTimeout);
                if constexpr (EXTERNAL_TRIGGER) SetWaitForTrigger(which_board, 1);
                machine_state = MACHINE_IDLE;
                break;

            case MACHINE_LOAD_LINEARMAP:
                loadLinearMatrix();
                machine_state = MACHINE_IDLE;
                break;


            default:
                std::cerr << "Unknown machine state: "<<machine_state<<"; terminating" << std::endl;
                machine_state = MACHINE_TERMINATING;
                break;
        }
        
    }

    /* free resources */

    return EXIT_SUCCESS;
}