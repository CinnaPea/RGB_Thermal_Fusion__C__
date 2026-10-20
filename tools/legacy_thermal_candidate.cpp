//
// Created by hakgu on 10/17/2026.
//

#include "legacy_thermal_candidate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <opencv2/ximgproc/edge_filter.hpp>

namespace qart::experiment {
namespace {

using namespace cv;

struct Scales_RGF
{
    int iters;
    int r;
    float eps;
    float sigma0;
};

Mat clamp(const Mat& image)
{
    Mat output;
    min(max(image, 0), 1, output);
    return output;
}

Mat normalize01(const Mat& image)
{
    double minimum;
    double maximum;
    minMaxLoc(image, &minimum, &maximum);

    if (maximum <= minimum + 1e-6) {
        return Mat::zeros(
            image.size(),
            CV_32F
        );
    }

    Mat values;
    image.convertTo(values, CV_32F);

    Mat output =
        (values - static_cast<float>(minimum))
        * (
            1.0f
            / static_cast<float>(maximum - minimum)
        );

    return output;
}

Mat local_entropy(
    const Mat& gray8,
    int window,
    int bins)
{
    CV_Assert(window > 1 && window % 2 == 1);

    Mat gray;

    if (gray8.type() != CV_8U) {
        gray8.convertTo(gray, CV_8U);
    }
    else {
        gray = gray8;
    }

    Mat quantized(gray.size(), CV_8U);

    for (int row = 0; row < gray.rows; ++row) {
        const auto* source =
            gray.ptr<std::uint8_t>(row);

        auto* destination =
            quantized.ptr<std::uint8_t>(row);

        for (
            int column = 0;
            column < gray.cols;
            ++column
        ) {
            destination[column] =
                static_cast<std::uint8_t>(
                    (source[column] * bins) >> 8
                );
        }
    }

    const int area = window * window;

    thread_local Mat lut(1, 256, CV_32F);
    thread_local int lastArea = -1;

    if (area != lastArea || lut.empty()) {
        lut.setTo(0);

        auto* values = lut.ptr<float>(0);

        const float inverseArea =
            1.0f / static_cast<float>(area);

        const float inverseLog2 =
            1.0f / std::log(2.0f);

        const int maximumCount =
            std::min(area, 255);

        for (
            int count = 1;
            count <= maximumCount;
            ++count
        ) {
            const float probability =
                static_cast<float>(count)
                * inverseArea;

            values[count] =
                -(
                    probability
                    * (
                        std::log(probability)
                        * inverseLog2
                    )
                );
        }

        lastArea = area;
    }

    Mat entropy = Mat::zeros(
        gray.size(),
        CV_32F
    );

    thread_local Mat mask;
    thread_local Mat count16;
    thread_local Mat count8;
    thread_local Mat contribution;

    for (int bin = 0; bin < bins; ++bin) {
        compare(
            quantized,
            bin,
            mask,
            CMP_EQ
        );

        boxFilter(
            mask,
            count16,
            CV_16U,
            Size(window, window),
            Point(-1, -1),
            false,
            BORDER_REFLECT101
        );

        count16 /= 255;

        count16.convertTo(
            count8,
            CV_8U
        );

        LUT(
            count8,
            lut,
            contribution
        );

        entropy += contribution;
    }

    entropy *= (
        1.0f
        / std::log2(static_cast<float>(bins))
    );

    min(max(entropy, 0), 1, entropy);

    return entropy;
}

Mat local_contrast(
    const Mat& gray8,
    int window)
{
    Mat gray;

    if (gray8.type() != CV_8U) {
        gray8.convertTo(gray, CV_8U);
    }
    else {
        gray = gray8;
    }

    Mat normalized;
    gray.convertTo(
        normalized,
        CV_32F,
        1.0 / 255.0
    );

    Mat mean;
    Mat meanSquared;

    boxFilter(
        normalized,
        mean,
        -1,
        Size(window, window),
        Point(-1, -1),
        true,
        BORDER_REFLECT101
    );

    boxFilter(
        normalized.mul(normalized),
        meanSquared,
        -1,
        Size(window, window),
        Point(-1, -1),
        true,
        BORDER_REFLECT101
    );

    Mat variance =
        meanSquared - mean.mul(mean);

    max(variance, 0, variance);

    Mat deviation;
    sqrt(variance, deviation);

    deviation *= 2.0f;

    return normalize01(deviation);
}

void weight_map(
    const Mat& thermal,
    Mat& fineOutput,
    Mat& structuralOutput)
{
    const Mat entropy =
        local_entropy(thermal, 7, 8);

    const Mat contrast =
        local_contrast(thermal, 7);

    const float contrastClip = 0.5f;

    Mat overload =
        (contrast - contrastClip)
        * (
            1.0f
            / (1.0f - contrastClip + 1e-6f)
        );

    min(max(overload, 0), 1, overload);

    Mat fine =
        1.0f
        + 0.8f * (entropy - 0.5f)
        - 0.5f * overload;

    Mat structural =
        1.0f
        + 0.6f * (entropy - 0.5f)
        + 0.6f * (contrast - 0.5f);

    Mat overloadGate =
        1.0f - 0.8f * overload;

    Mat contrastGate =
        0.35f + 0.65f * contrast;

    fine = fine.mul(overloadGate);
    fine = fine.mul(contrastGate);

    min(max(fine, 0.3f), 1.0f, fine);

    min(
        max(structural, 0.8f),
        1.4f,
        structural
    );

    fineOutput = fine;
    structuralOutput = structural;
}

Mat guided_filter_gray(
    const Mat& guide,
    const Mat& source,
    int radius,
    float epsilon)
{
    Mat output;

    ximgproc::guidedFilter(
        guide,
        source,
        output,
        radius,
        epsilon
    );

    min(max(output, 0), 1, output);

    return output;
}

Mat rolling_guided_filter(
    const Mat& thermal,
    int iterations,
    int radius,
    float epsilon,
    float initialSigma)
{
    CV_Assert(thermal.type() == CV_8U);

    Mat source;
    thermal.convertTo(
        source,
        CV_32F,
        1.0 / 255.0
    );

    Mat guide;
    GaussianBlur(
        source,
        guide,
        Size(0, 0),
        initialSigma,
        initialSigma,
        BORDER_REFLECT101
    );

    Mat base = guide.clone();

    const int count =
        std::max(1, iterations);

    if (iterations <= 1) {
        base = guided_filter_gray(
            guide,
            source,
            radius,
            epsilon
        );
    }
    else {
        for (int index = 0; index < count; ++index) {
            base = guided_filter_gray(
                guide,
                source,
                radius,
                epsilon
            );

            guide = base;
        }
    }

    return base;
}

std::vector<Mat> rgf_multi_scale(
    const Mat& image,
    const std::vector<Scales_RGF>& scales)
{
    std::vector<Mat> bases;
    bases.reserve(scales.size());

    for (
        const auto& [iterations, radius, epsilon, sigma]
        : scales
    ) {
        Mat base = rolling_guided_filter(
            image,
            iterations,
            radius,
            epsilon,
            sigma
        );

        bases.push_back(base);
    }

    return bases;
}

void rgf_decompose(
    const Mat& thermal,
    const std::vector<Scales_RGF>& scales,
    Mat& fine,
    Mat& medium,
    Mat& large,
    Mat& base)
{
    if (scales.size() != 3) {
        throw std::runtime_error(
            "Legacy candidate requires three scales"
        );
    }

    Mat source;
    thermal.convertTo(
        source,
        CV_32F,
        1.0 / 255.0
    );

    const auto bases =
        rgf_multi_scale(thermal, scales);

    const Mat first = bases[0];
    const Mat second = bases[1];
    const Mat third = bases[2];

    fine = source - first;

    min(
        max(fine, -0.15f),
        0.15f,
        fine
    );

    medium = first - second;
    large = second - third;
    base = third;
}

void enhance_details(
    const Mat& fine,
    const Mat& medium,
    const Mat& large,
    const Mat& fineWeights,
    const Mat& structuralWeights,
    Mat& fineOutput,
    Mat& structuralOutput)
{
    const Mat structural =
        medium + large;

    fineOutput =
        fine.mul(fineWeights);

    min(
        max(fineOutput, -0.07f),
        0.07f,
        fineOutput
    );

    structuralOutput =
        structural.mul(structuralWeights);
}

Mat build_luminance(
    const Mat& fine,
    const Mat& structural,
    const Mat& mappedBase,
    float fineCoefficient,
    float structuralCoefficient,
    float baseCoefficient,
    float fineClip)
{
    Mat clippedFine;
    fine.copyTo(clippedFine);

    min(
        max(clippedFine, -fineClip),
        fineClip,
        clippedFine
    );

    Mat output =
        baseCoefficient * mappedBase
        + fineCoefficient * clippedFine
        + structuralCoefficient * structural;

    output = clamp(output);

    return output;
}

} // namespace

LegacyThermalCandidate::LegacyThermalCandidate()
    : clahe_(
        cv::createCLAHE(
            2.0,
            cv::Size(8, 8)
        )
    )
{
}

    cv::Mat LegacyThermalCandidate::enhance(
        const cv::Mat& plane,
        const std::string& tracePath)
{
    if (
        plane.empty()
        || plane.dims != 2
        || plane.type() != CV_8UC1
    ) {
        throw std::invalid_argument(
            "Legacy candidate requires a nonempty "
            "2D CV_8UC1 thermal plane"
        );
    }

    const std::vector<Scales_RGF> scales{
        {1, 4, 1e-3f, 1.2f},
        {1, 4, 1e-3f, 2.0f},
        {1, 4, 1e-3f, 3.0f}
    };

    cv::Mat fine;
    cv::Mat medium;
    cv::Mat large;
    cv::Mat base;

    rgf_decompose(
        plane,
        scales,
        fine,
        medium,
        large,
        base
    );

    cv::Mat fineWeights;
    cv::Mat structuralWeights;

    weight_map(
        plane,
        fineWeights,
        structuralWeights
    );

    cv::GaussianBlur(
        fineWeights,
        fineWeights,
        cv::Size(5, 5),
        1.0, 1.0,
        cv::BORDER_REFLECT101
    );

    cv::GaussianBlur(
        structuralWeights,
        structuralWeights,
        cv::Size(5, 5),
        1.0, 1.0,
        cv::BORDER_REFLECT101
    );

    cv::Mat enhancedFine;
    cv::Mat enhancedStructure;

    enhance_details(
        fine,
        medium,
        large,
        fineWeights,
        structuralWeights,
        enhancedFine,
        enhancedStructure
    );

    const cv::Mat scaledBase =
        base * 255.0f;

    cv::Mat base8;
    scaledBase.convertTo(
        base8,
        CV_8U
    );

    cv::Mat mapped8;
    clahe_->apply(base8, mapped8);

    cv::Mat mappedBase;
    mapped8.convertTo(
        mappedBase,
        CV_32F,
        1.0 / 255.0
    );

    const cv::Mat reconstructed =
        build_luminance(
            enhancedFine,
            enhancedStructure,
            mappedBase,
            1.0f,
            1.0f,
            1.0f,
            0.07f
        );

    cv::Mat output;
    reconstructed.convertTo(
        output,
        CV_8U,
        255.0
    );

    if (!tracePath.empty()) {
        cv::FileStorage trace(
            tracePath,
            cv::FileStorage::WRITE
        );

        if (!trace.isOpened()) {
            throw std::runtime_error(
                "Cannot write trace: " + tracePath
            );
        }

        cv::Mat normalizedSource;
        plane.convertTo(
            normalizedSource,
            CV_32F,
            1.0 / 255.0
        );

        trace
            << "source" << normalizedSource
            << "detail_fine" << fine
            << "detail_medium" << medium
            << "detail_large" << large
            << "base" << base
            << "fine_weights" << fineWeights
            << "structural_weights" << structuralWeights
            << "enhanced_fine" << enhancedFine
            << "enhanced_structure" << enhancedStructure
            << "scaled_base" << scaledBase
            << "base8" << base8
            << "mapped8" << mapped8
            << "mapped_base" << mappedBase
            << "reconstructed" << reconstructed
            << "output" << output;

        trace.release();
    }
    return output;
}

} // namespace qart::experiment