//
// Created by tujun on 2026/8/20.
//

#ifndef SLM_HOLOGRAM_CONFIGURATIONS_H
#define SLM_HOLOGRAM_CONFIGURATIONS_H

// NON-CONFIGURABLE PARAMETERS
#define SLM_TYPE_532    0
#define SLM_TYPE_840    1
#define SLM_TYPE_BOTH   2
#define FLAG_NULL       0
#define FLAG_NEW        1
#define FLAG_EVOLVE     2
#define GS_MODE_DIRECT  0
#define GS_MODE_SELECTED 1
#define GS_MODE_WEIGHTED 2
#define GS_MODE_SELECTED_WEIGHTED 3
#define MAX_TWEEZER_NUM 4096
#define MACHINE_IDLE 0
#define MACHINE_GENERATING 1
#define MACHINE_DUMMY 2
#define MACHINE_ZERO 3
#define MACHINE_POSITION_CALIBRATING 4
#define MACHINE_POSITION_CALIBRATING_ARMED 5
#define MACHINE_INTENSITY_CALIBRATING 6
#define MACHINE_INTENSITY_CALIBRATING_ARMED 7
#define MACHINE_SAVE_HOLOGRAM 8
#define MACHINE_LOAD_HOLOGRAM 9
#define MACHINE_LOAD_LINEARMAP 10
#define MACHINE_TERMINATING -1
#define MACHINE_TERMINATED -2
#define BUFFER_SIZE 65536

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OPTICAL PARAMETERS
#define PIXEL_WIDTH 1024
#define PIXEL_HEIGHT 1024
#define PIXEL_PITCH 17.f // in um
#define BEAM_RADIUS_532 514.f // in SLM pixels
#define BEAM_RADIUS_840 514.f // in SLM pixels
#define WFC_FILE_PATH_532 "C:\\Program Files\\Meadowlark Optics\\Blink Plus\\WFC Files\\slm6889_at532_58C.bmp"
#define WFC_FILE_PATH_840 "C:\\Program Files\\Meadowlark Optics\\Blink Plus\\WFC Files\\slm7598_at785_75C_WFC.bmp"
#define FOCAL_LENGTH 30000.f // in um; objective focal length 30 mm
#define MAGNIFICATION_532 1.875f // magnification 1.875 (Sarthak's thesis p149)
#define MAGNIFICATION_840 1.875f // magnification 1.875 (Sarthak's thesis p149)
#define WAVELENGTH_532 532e-3f // in um
#define WAVELENGTH_840 840e-3f // in um

// SOFTWARE PARAMETERS
#define MAX_ITERATION 10 // maximum number of GS iterations per hologram
#define IF_GS_THRESHOLD 1 // 1 for using GS threshold, 0 for not and iterating maximum number
#define GS_THRESHOLD 0.97f // The threshold for satisfying GS convergence; between 0 and 1
#define TRIGGER_TIMEOUT 5000 // in ms
#define SLM_TYPE SLM_TYPE_BOTH // SLM_TYPE_532, SLM_TYPE_840, or SLM_TYPE_BOTH
#define GS_MODE GS_MODE_DIRECT // DIRECT, SELECTED, WEIGHTED, or SELECTED_WEIGHTED
#define EXTERNAL_TRIGGER 0 // 1 for external trigger, 0 for no trigger
#define FLIP_IMMEDIATE 0 // 1 for flip immediately, 0 for flip after full update
#define OUTPUT_PULSE 0 // 1 for output pulse, 0 for no output pulse

// DUMMY TEST PARAMETERS
#define OFFSET_X_532 491 // The left corner of the dummy test area in the 532 SLM image
#define OFFSET_Y_532 491 // The top corner of the dummy test area in the 532 SLM image
#define OFFSET_X_840 491 // The left corner of the dummy test area in the 840 SLM image
#define OFFSET_Y_840 491 // The top corner of the dummy test area in the 840 SLM image
#define DUMMY_PITCH_532 10 // in pixel
#define DUMMY_PITCH_840 10 // in pixel
#define DUMMY_SIDE_532 5 // number of tweezer per side
#define DUMMY_SIDE_840 5 // number of tweezer per side

// Image processing/calibration parameters
#define SPOT_RADIUS 0.5 //  in pixels, estimated 1/e^2 radius of each spot
#define BACKGROUND_INNER_RADIUS 0.5 // in pixels, the inner radius of local background sampling annulus
#define BACKGROUND_OUTER_RADIUS 0.5 // in pixels, the outer radius of local background sampling annulus
#define SPOT_MIN_DISTANCE 0.5 // in pixels, the minimum distance between two spots, used to avoid double counting
#define DISTINGUISH_THRESHOLD 0.5 // in intensity distributiuon std, the threshold = mean + std * DISTINGUISH_THRESHOLD
#define IF_X_INVERTED 0 // 1 for x inverted, 0 for not, compared with location programmed
#define IF_Y_INVERTED 0 // 1 for x inverted, 0 for not, compared with location programmed
#define HOMOGENEITY_THRESHOLD 0.03 // the std/mean intensity threshold for satisfying homogeneity
#define CALIBRATION_FILENAME "pixel_mapper"


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NON-CONFIGURABLE PARAMETERS

constexpr float Fourier_pitch_532 = WAVELENGTH_532 * FOCAL_LENGTH / PIXEL_HEIGHT / MAGNIFICATION_532 / PIXEL_PITCH; // in um, 0.489
constexpr float Fourier_pitch_840 = WAVELENGTH_840 * FOCAL_LENGTH / PIXEL_HEIGHT / MAGNIFICATION_840 / PIXEL_PITCH; // in um, 0.772

enum class SLMType : int {
    SLM_532  = SLM_TYPE_532,
    SLM_840  = SLM_TYPE_840,
    SLM_both = SLM_TYPE_BOTH,
};

constexpr SLMType SLM_type = static_cast<SLMType>(SLM_TYPE);


#endif //SLM_HOLOGRAM_CONFIGURATIONS_H
