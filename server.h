#ifndef _server_included_
#define _server_included_

#include <winsock2.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <sys/types.h>
#include <math.h>
#include <errno.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include "CONFIGURATIONS.h"
#include <ws2tcpip.h>


constexpr int nx = PIXEL_WIDTH;
constexpr int ny = PIXEL_HEIGHT;
constexpr int total_pixel = nx * ny;

struct hologram_params {    
    const float wavelength;
    const float beam_radius;
    int board_index = 0;
    int ptnum = 0;
    int mux[MAX_TWEEZER_NUM] = {0};
    int muy[MAX_TWEEZER_NUM] = {0};
    float amp[MAX_TWEEZER_NUM] = {1.f};
    int flag = FLAG_NULL;
    bool is_defocus = false;
    float defocus = 0.f;
    float beam_amp[total_pixel];
    float target_amp[total_pixel];
    unsigned char WFCimage[total_pixel];
    half* beam_amp_cuda;
    half* target_amp_cuda;
    half* oamp_cuda;
};

extern hologram_params holo_params_532;
extern hologram_params holo_params_840;

extern volatile int machine_state;

extern int target_mux[MAX_TWEEZER_NUM];
extern int target_muy[MAX_TWEEZER_NUM];
extern int dynamic_index[MAX_TWEEZER_NUM];
extern int dynamic_num;
extern cv::Mat image_buffer;
extern char * phase_mask_name;


extern int TCP_server();

#endif