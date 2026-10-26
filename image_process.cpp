#include "image_process.hpp"

OptionalCVMatx23f positionMatrix;

static float medianOf(std::vector<float>& values) {
    if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
    const size_t n = values.size();
    auto middle = values.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(values.begin(), middle, values.end());
    const float upper = *middle;
    if (n & 1U) return upper;
    auto lowerIt = std::max_element(values.begin(), middle);
    return 0.5 * (upper + *lowerIt);
}

static float robustSigma(const cv::Mat& image) {
    std::vector<float> values;
    values.reserve(image.total());
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            values.push_back(image.at<float>(y, x));
    const float med = medianOf(values);
    for (float& value : values) value = std::abs(value - med);
    return 1.4826 * medianOf(values);
}

static std::vector<float> matToVector(const cv::Mat& image) {
    std::vector<float> values;
    values.reserve(image.total());

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            values.push_back(image.at<float>(y, x));
        }
    }

    return values;
}

static std::vector<cv::Point2f> findCandidates(const cv::Mat& raw, const SpotDetectorConfig& cfg,
                                               float& thresholdOut) {
    cv::Mat response;
    cv::GaussianBlur(raw, response, cv::Size(), cfg.spotSigma, cfg.spotSigma, cv::BORDER_REPLICATE);
    cv::Laplacian(response, response, CV_32F, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    response *= - cfg.spotSigma * cfg.spotSigma;

    const float median = medianOf(matToVector(response));
    thresholdOut = median + cfg.thresholdSigma * (std::max)(robustSigma(response), 1e-12f);

    std::vector<cv::Point2f> allMaxima;
    const int radius = (std::max)(1, static_cast<int>(std::ceil(cfg.minDistance)));
    for (int y = radius; y < response.rows - radius; ++y) {
        for (int x = radius; x < response.cols - radius; ++x) {
            const float center = response.at<float>(y, x);
            if (center <= thresholdOut) continue;
            bool isMaximum = true;
            for (int yy = y - radius; yy <= y + radius && isMaximum; ++yy)
                for (int xx = x - radius; xx <= x + radius; ++xx)
                    if ((xx != x || yy != y) && response.at<float>(yy, xx) > center) {
                        isMaximum = false;
                        break;
                    }
            if (isMaximum) allMaxima.emplace_back(x, y);
        }
    }

    // Resolve equal-value plateaus and ensure only one candidate within minDistance.
    std::sort(allMaxima.begin(), allMaxima.end(), [&](const cv::Point2f& a, const cv::Point2f& b) {
        return response.at<float>(a.y, a.x) > response.at<float>(b.y, b.x);
    });
    std::vector<cv::Point2f> accepted;
    const float minDistance2 = cfg.minDistance * cfg.minDistance;
    for (const auto& point : allMaxima) {
        bool farEnough = true;
        for (const auto& previous : accepted) {
            const float dx = point.x - previous.x, dy = point.y - previous.y;
            if (dx * dx + dy * dy < minDistance2) { farEnough = false; break; }
        }
        if (farEnough) accepted.push_back(point);
    }
    return accepted;
}


static bool measureCandidate(const cv::Mat& raw, const cv::Point2f& candidate,
                             const SpotDetectorConfig& cfg, SpotMeasurement& output) {
    const int margin = static_cast<int>(std::ceil(cfg.annulusOuterRadius));
    const int left = (std::max)(0.f, candidate.x - margin);
    const int right = (std::min)(raw.cols - 1.f, candidate.x + margin);
    const int top = (std::max)(0.f, candidate.y - margin);
    const int bottom = (std::min)(raw.rows - 1.f, candidate.y + margin);

    std::vector<float> annulusValues;
    for (int y = top; y <= bottom; ++y) for (int x = left; x <= right; ++x) {
        const float dx = x - candidate.x, dy = y - candidate.y;
        const float r2 = dx * dx + dy * dy;
        if (r2 >= cfg.annulusInnerRadius * cfg.annulusInnerRadius &&
            r2 <= cfg.annulusOuterRadius * cfg.annulusOuterRadius)
            annulusValues.push_back(raw.at<float>(y, x));
    }
    if (annulusValues.size() < 5) return false;
    const float background = medianOf(annulusValues);

    float sum = 0.0, weightedX = 0.0, weightedY = 0.0, peak = 0.0;
    const float aperture2 = cfg.apertureRadius * cfg.apertureRadius;
    for (int y = top; y <= bottom; ++y) for (int x = left; x <= right; ++x) {
        const float dx = x - candidate.x, dy = y - candidate.y;
        if (dx * dx > aperture2 || dy * dy > aperture2 || dx * dx + dy * dy > aperture2) continue;
        float signal = raw.at<float>(y, x) - background;
        signal = (std::max)(0.0f, signal);
        sum += signal; weightedX += x * signal; weightedY += y * signal; peak = (std::max)(peak, signal);
    }
    if (sum <= 0.0) return false;

    output.cameraXpx = weightedX / sum;
    output.cameraYpx = weightedY / sum;
    output.localBackgroundAdu = background;
    output.peakIntensityAdu = peak;

    // Recenter the photometry aperture on the subpixel centroid.
    output.integratedIntensityAdu = 0.0;
    output.aperturePixelCount = 0;
    for (int y = top; y <= bottom; ++y) for (int x = left; x <= right; ++x) {
        const float dx = x - output.cameraXpx, dy = y - output.cameraYpx;
        if (dx * dx + dy * dy > aperture2) continue;
        float signal = raw.at<float>(y, x) - background;
        output.integratedIntensityAdu += (std::max)(0.0f, signal);
        ++output.aperturePixelCount;
    }
    return true;
}

static SpotResults detectAndMeasureSpots(const cv::Mat& cameraFrame, const SpotDetectorConfig& cfg) {
    const cv::Mat raw;
    cameraFrame.convertTo(raw, CV_32F);
    SpotResults results;
    const auto candidates = findCandidates(raw, cfg, results.detectionThreshold);
    results.candidateCount = static_cast<int>(candidates.size());
    for (const auto& candidate : candidates) {
        SpotMeasurement spot;
        if (measureCandidate(raw, candidate, cfg, spot)) results.spots.push_back(spot);
    }
    std::sort(results.spots.begin(), results.spots.end(), [](const auto& a, const auto& b) {
        return a.cameraYpx != b.cameraYpx ? a.cameraYpx < b.cameraYpx : a.cameraXpx < b.cameraXpx;
    });
    for (size_t i = 0; i < results.spots.size(); ++i) results.spots[i].spotId = static_cast<int>(i);
    return results;
}


static AffineFit fitAffine2x3(
    const std::vector<cv::Point2f>& programmed,
    const std::vector<cv::Point2f>& measured
) {
    const int n = static_cast<int>(programmed.size());

    // Design matrix:
    //
    // X = [ programmed_x  programmed_y  1 ]
    //
    // Solve X * B = measured, where B is 3x2.
    cv::Mat X(n, 3, CV_32F);
    cv::Mat Y(n, 2, CV_32F);

    for (int i = 0; i < n; ++i) {
        X.at<float>(i, 0) = programmed[i].x;
        X.at<float>(i, 1) = programmed[i].y;
        X.at<float>(i, 2) = 1.0;

        Y.at<float>(i, 0) = measured[i].x;
        Y.at<float>(i, 1) = measured[i].y;
    }

    cv::Mat B;

    if (!cv::solve(X, Y, B, cv::DECOMP_SVD)) {
        throw std::runtime_error("Could not solve affine least-squares fit");
    }

    // B is 3x2; transpose its layout into conventional 2x3 form.
    cv::Matx23f A(
        B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0),
        B.at<float>(0, 1), B.at<float>(1, 1), B.at<float>(2, 1)
    );

    float squaredErrorSum = 0.0;
    float maxError = 0.0;

    for (int i = 0; i < n; ++i) {
        const cv::Point2f& p = programmed[i];

        const cv::Point2f predicted(
            A(0, 0) * p.x + A(0, 1) * p.y + A(0, 2),
            A(1, 0) * p.x + A(1, 1) * p.y + A(1, 2)
        );

        const float error = cv::norm(predicted - measured[i]);

        squaredErrorSum += error * error;
        maxError = (std::max)(maxError, error);
    }

    return {
        A,
        std::sqrt(squaredErrorSum / n),
        maxError
    };
}


static void saveAffineMatrix(
    const cv::Matx23f& A
) {
    cv::FileStorage file(CALIBRATION_FILENAME, cv::FileStorage::WRITE);

    if (!file.isOpened()) {
        std::cerr << "Could not open file for writing: " << CALIBRATION_FILENAME << std::endl;
        throw std::runtime_error("saveAffineMatrix failed");
    }

    // Save a matrix rather than six unrelated values.
    cv::Mat matrix(A);

    file << "model" << "affine_2x3";
    file << "direction" << "programmed_to_camera_pixels";
    file << "A" << matrix;

    file.release();
}





static int map_calibrator(const SpotResults& measuredSpots, const int ptnum, const int* mux, const int* muy) {
    std::vector <cv::Point2f> vprogrammedSpots;
    for (int i = 0; i < ptnum; i++) {
        vprogrammedSpots.emplace_back(mux[i], muy[i]);
    }
    if (measuredSpots.spots.size() != ptnum) {
        std::cerr << "Error: Number of measured spots does not match programmed spots" << '\n';
        return 1;
    }
    std::vector <cv::Point2f> vMeasuredSpots;
    for (int i = 0; i < ptnum; i++) {
        vMeasuredSpots.emplace_back(measuredSpots.spots[i].cameraXpx, measuredSpots.spots[i].cameraYpx);
    }
    std::sort(vprogrammedSpots.begin(), vprogrammedSpots.end(),
    [](const auto& a, const auto& b) {
        const int ax = a.x;
        const int ay = a.y;

        const int bx = b.x;
        const int by = b.y;
        if (ax != bx) {
            return ax < bx;   // x ascending
        }
        return ay < by;   // y ascending
    });
    std::sort(vMeasuredSpots.begin(), vMeasuredSpots.end(),
    [](const auto& a, const auto& b) {
        const int ax = a.x;
        const int ay = a.y;

        const int bx = b.x;
        const int by = b.y;
        if (ax != bx) {
            if constexpr (IF_X_INVERTED){
                return ax > bx;   // x descending
            }else{
                return ax < bx;   // x ascending
            }
        }
        if constexpr (IF_Y_INVERTED){
            return ay > by;   // y descending
        }else{
            return ay < by;   // y ascending
        }
    });

    AffineFit fit = fitAffine2x3(vprogrammedSpots, vMeasuredSpots);

    std::cout<< "Affine fit: " << fit.A << '\n';
    std::cout<< "Affine fit error: " << fit.rmsErrorPx << '\n';
    std::cout<< "Affine fit max error: " << fit.maxErrorPx << '\n';
    
    positionMatrix.value = fit.A;
    positionMatrix.empty = false;

    saveAffineMatrix(positionMatrix.value);
    return 0;
}



int loadLinearMatrix() {
    cv::FileStorage file(CALIBRATION_FILENAME, cv::FileStorage::READ);

    if (!file.isOpened()) {
        std::cerr << "Cannot read: " << CALIBRATION_FILENAME << std::endl;
        return 1;
    }

    std::string model;
    cv::Mat matrix;

    file["model"] >> model;
    file["A"] >> matrix;

    file.release();

    if (model != "linear_2x3" ||
        matrix.rows != 2 ||
        matrix.cols != 3) {
        throw std::runtime_error(
            "File does not contain a valid 2x3 linear matrix"
        );
    }

    matrix.convertTo(matrix, CV_32F);

    positionMatrix.value = cv::Matx23f(
        matrix.at<float>(0, 0),
        matrix.at<float>(0, 1),
        matrix.at<float>(0, 2),
        matrix.at<float>(1, 0),
        matrix.at<float>(1, 1),
        matrix.at<float>(1, 2)
    );
    positionMatrix.empty = false;

    return 0;
}



static int intensity_calibrator(const cv::Mat& raw, const SpotDetectorConfig& cfg, const int ptnum, const int* mux, const int* muy, float* amp) {
    if (positionMatrix.empty) {
        if (loadLinearMatrix()) {
            std::cerr << "Lack of position map. Do position map loading or calibration first." << '\n';
            return -1;
        }    
    }

    cv::Mat mProgrammedSpots(3, ptnum, CV_32F);
    for (int i = 0; i < ptnum; i++) {
        mProgrammedSpots.at<float>(0, i) = mux[i];
        mProgrammedSpots.at<float>(1, i) = muy[i];
        mProgrammedSpots.at<float>(2, i) = 1.0;
    }
    cv::Mat flattenExpectedSpots = positionMatrix.value * mProgrammedSpots;
    // std::vector<cv::Point2f> expectedPoints;
    // flattenExpectedSpots.copyTo(expectedPoints);
    std::vector<float> intensityResults;
    float avg = 0.f;
    float std = 0.f;
    for (int i = 0; i < ptnum; i++) {
        SpotMeasurement spot;
        if (!measureCandidate(raw, cv::Point2f(flattenExpectedSpots.at<float>(0, i), flattenExpectedSpots.at<float>(1, i)),
            cfg, spot)) {
                std::cerr << "Error: Failed to measure candidate" << '\n';
                return 1;
            }
        intensityResults.push_back(spot.peakIntensityAdu);
        avg += spot.peakIntensityAdu;
    }
    avg /= ptnum;
    for (int i = 0; i < ptnum; i++) {
        float diff = intensityResults[i] - avg;
        std += diff * diff;
    }
    std = sqrt(std / ptnum) / avg;
    if (std < HOMOGENEITY_THRESHOLD) {
        return 0;    
    }
    float powersum = 0;
    for (int i = 0; i < ptnum; i++) {
        amp[i] /= sqrt(intensityResults[i]);
        powersum += amp[i]*amp[i];
    }
    for (int i = 0; i < ptnum; i++) {
        amp[i] /= sqrt(powersum);
    }
    return 1;
}



const SpotDetectorConfig kSpotDetectorConfig{};


int positionCalibration(const cv::Mat& image, const hologram_params& params) {
    const SpotDetectorConfig& cfg = kSpotDetectorConfig;
    if (!(cfg.apertureRadius > 0.0 && cfg.apertureRadius < cfg.annulusInnerRadius &&
          cfg.annulusInnerRadius < cfg.annulusOuterRadius))
        throw std::invalid_argument("Require 0 < apertureRadius < annulusInnerRadius < annulusOuterRadius");
    if (image.empty()) {
        std::cerr << "Error: No image data received" << '\n';
        return 1;
    }
    try {
        const SpotResults measurements = detectAndMeasureSpots(image, cfg);
        std::cout << "Candidates: " << measurements.candidateCount << "\nMeasured spots: "
                  << measurements.spots.size() << "\nThreshold: " << measurements.detectionThreshold << "\n";

        if (map_calibrator(measurements, params.ptnum, params.mux, params.muy)){
            throw std::runtime_error("Failed to map and save calibration");
        };
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}





int intensityCalibration(const cv::Mat& image, hologram_params& params) {
    const SpotDetectorConfig& cfg = kSpotDetectorConfig;
    if (!(cfg.apertureRadius > 0.0 && cfg.apertureRadius < cfg.annulusInnerRadius &&
          cfg.annulusInnerRadius < cfg.annulusOuterRadius))
        throw std::invalid_argument("Require 0 < apertureRadius < annulusInnerRadius < annulusOuterRadius");
    if (image.empty()) {
        std::cerr << "Error: No image data received" << '\n';
        return 1;
    }
    try {
        const int result = intensity_calibrator(image, cfg, params.ptnum, params.mux, params.muy, params.amp);
        if (result == -1){
            std::cerr << "Error: failed to calibrate intensity" << std::endl;
            return 1;
        }else if (result == 1){
            std::cout << "Not enough homogeneity; new weights applied and new calibration cycle needed" << std::endl;
            return 2;
        }else if (result == 0){
            std::cout << "Calibration successful" << std::endl;
        }else{
            throw std::runtime_error("Unknown behavior at intensity calibration");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
