#ifndef _IMAGE_PROCESS_INCLUDED_
#define _IMAGE_PROCESS_INCLUDED_
#include "CONFIGURATIONS.h"
#include "server.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <string>

struct OptionalCVMatx23f {
    bool empty = true;
    cv::Matx23f value;
};

struct AffineFit {
    cv::Matx23f A;       // measured = A * [programmed_x, programmed_y, 1]
    float rmsErrorPx;
    float maxErrorPx;
};


struct SpotMeasurement {
    int spotId = -1;
    float cameraXpx = 0.0;             // intensity-weighted subpixel centroid
    float cameraYpx = 0.0;
    float integratedIntensityAdu = 0.0; // local-background-subtracted aperture sum
    float peakIntensityAdu = 0.0;      // background-subtracted peak in the aperture
    float localBackgroundAdu = 0.0;    // median of annulus pixels
    int aperturePixelCount = 0;
};

struct SpotResults {
    std::vector<SpotMeasurement> spots; // The requested measurement array.
    int candidateCount = 0;
    float detectionThreshold = 0.0;
};

struct SpotDetectorConfig {
    float spotSigma = SPOT_RADIUS;
    float thresholdSigma = DISTINGUISH_THRESHOLD;
    float minDistance = SPOT_MIN_DISTANCE;
    float apertureRadius = 3. * spotSigma;
    float annulusInnerRadius = BACKGROUND_INNER_RADIUS;
    float annulusOuterRadius = BACKGROUND_OUTER_RADIUS;
};


extern int positionCalibration(const cv::Mat& image, const hologram_params& params);
extern int intensityCalibration(const cv::Mat& image, hologram_params& params);
extern int loadLinearMatrix();

#endif