//
// Created by hakgu on 10/16/2026.
//

#include "n5_thermal_enhancer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <chrono>

namespace qart::experiment {

    using qart::core::enhancement::N5EnhancementResult;
    using qart::core::enhancement::N5Stage;
    using qart::core::enhancement::N5StageTimings;

namespace {

struct GuidedBaseResult
{
    cv::Mat image;
    double regularization;
};

void validateInput(const cv::Mat& plane)
{
    if (
        plane.empty()
        || plane.dims != 2
        || plane.type() != CV_8UC1
    ) {
        throw std::invalid_argument(
            "N5 requires a nonempty 2D CV_8UC1 thermal plane"
        );
    }
}

cv::Mat normalizeUint8(const cv::Mat& input)
{
    cv::Mat output(input.rows, input.cols, CV_64FC1);

    for (int row = 0; row < input.rows; ++row) {
        const auto* source = input.ptr<unsigned char>(row);
        auto* destination = output.ptr<double>(row);

        for (int column = 0; column < input.cols; ++column) {
            destination[column] =
                static_cast<double>(source[column]) / 255.0;
        }
    }

    return output;
}

// Match np.rint for the nonnegative, clipped output range.
// Avoid std::round, which rounds halfway values away from zero.
unsigned char quantizeValue(double value)
{
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "N5 reconstruction contains a nonfinite value"
        );
    }

    const double scaled =
        std::clamp(value, 0.0, 1.0) * 255.0;

    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;

    int rounded = static_cast<int>(lower);

    if (
        fraction > 0.5
        || (fraction == 0.5 && rounded % 2 != 0)
    ) {
        ++rounded;
    }

    return static_cast<unsigned char>(rounded);
}

cv::Mat quantizeUint8(const cv::Mat& input)
{
    cv::Mat output(input.rows, input.cols, CV_8UC1);

    for (int row = 0; row < input.rows; ++row) {
        const auto* source = input.ptr<double>(row);
        auto* destination = output.ptr<unsigned char>(row);

        for (int column = 0; column < input.cols; ++column) {
            destination[column] = quantizeValue(source[column]);
        }
    }

    return output;
}

cv::Mat boxMean(const cv::Mat& input, int window)
{
    cv::Mat output;

    cv::boxFilter(
        input,
        output,
        CV_64F,
        cv::Size(window, window),
        cv::Point(-1, -1),
        true,
        cv::BORDER_REFLECT_101
    );

    return output;
}

// Fixed guided-base definition from the frozen Python reference:
// mu = M(x)
// variance = max(M(x*x) - mu*mu, 0)
// epsilon = mean(variance)
// a = variance / (variance + epsilon), zero when denominator is zero
// base = M(a)*x + M((1-a)*mu)
GuidedBaseResult guidedBase(
    const cv::Mat& source,
    int window)
{
    const cv::Mat mean = boxMean(source, window);

    cv::Mat squared(source.rows, source.cols, CV_64FC1);

    for (int row = 0; row < source.rows; ++row) {
        const auto* input = source.ptr<double>(row);
        auto* output = squared.ptr<double>(row);

        for (int column = 0; column < source.cols; ++column) {
            output[column] = input[column] * input[column];
        }
    }

    const cv::Mat meanSquared = boxMean(squared, window);
    cv::Mat variance(source.rows, source.cols, CV_64FC1);

    for (int row = 0; row < source.rows; ++row) {
        const auto* mu = mean.ptr<double>(row);
        const auto* averageSquare = meanSquared.ptr<double>(row);
        auto* output = variance.ptr<double>(row);

        for (int column = 0; column < source.cols; ++column) {
            const double squareOfMean =
                mu[column] * mu[column];

            output[column] = std::max(
                averageSquare[column] - squareOfMean,
                0.0
            );
        }
    }

    const double epsilon = cv::mean(variance)[0];

    if (!std::isfinite(epsilon)) {
        throw std::runtime_error(
            "N5 guided regularization is nonfinite"
        );
    }

    cv::Mat coefficient(source.rows, source.cols, CV_64FC1);
    cv::Mat intercept(source.rows, source.cols, CV_64FC1);

    for (int row = 0; row < source.rows; ++row) {
        const auto* localVariance = variance.ptr<double>(row);
        const auto* mu = mean.ptr<double>(row);
        auto* a = coefficient.ptr<double>(row);
        auto* b = intercept.ptr<double>(row);

        for (int column = 0; column < source.cols; ++column) {
            const double denominator =
                localVariance[column] + epsilon;

            a[column] = denominator > 0.0
                ? localVariance[column] / denominator
                : 0.0;

            b[column] = (1.0 - a[column]) * mu[column];
        }
    }

    const cv::Mat averagedCoefficient =
        boxMean(coefficient, window);

    const cv::Mat averagedIntercept =
        boxMean(intercept, window);

    cv::Mat base(source.rows, source.cols, CV_64FC1);

    for (int row = 0; row < source.rows; ++row) {
        const auto* input = source.ptr<double>(row);
        const auto* a = averagedCoefficient.ptr<double>(row);
        const auto* b = averagedIntercept.ptr<double>(row);
        auto* output = base.ptr<double>(row);

        for (int column = 0; column < source.cols; ++column) {
            const double product = a[column] * input[column];
            output[column] = product + b[column];
        }
    }

    return {std::move(base), epsilon};
}

    class StageClock final
    {
    public:
        explicit StageClock(N5StageTimings* timings)
            : timings_(timings)
        {
            if (timings_) {
                timings_->milliseconds.fill(0.0);
                previous_ = Clock::now();
            }
        }

        void mark(N5Stage stage)
        {
            if (!timings_) {
                return;
            }

            const auto now = Clock::now();

            timings_->milliseconds[
                static_cast<std::size_t>(stage)
            ] = std::chrono::duration<double, std::milli>(
                now - previous_
            ).count();

            previous_ = now;
        }

    private:
        using Clock = std::chrono::steady_clock;

        N5StageTimings* timings_;
        Clock::time_point previous_{};
    };
} // namespace

    N5EnhancementResult enhance(
        const cv::Mat& plane,
        const cv::Mat& correction)
{
    StageClock profile(nullptr);
    validateInput(plane);

    if (
        correction.dims != 2
        || correction.type() != CV_64FC1
        || correction.size() != plane.size()
    ) {
        throw std::invalid_argument(
            "Experimental N5 correction must be matching CV_64FC1"
        );
    }

    auto clahe_ = cv::createCLAHE(
        2.0,
        cv::Size(8, 8)
    );

    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(plane, &minimum, &maximum);

    const bool constant = minimum == maximum;
    cv::Mat unitUint8;
    profile.mark(N5Stage::InputValidation);

    if (constant) {
        // UNIT preserves constant inputs before its CLAHE stage.
        unitUint8 = plane.clone();
    }
    else {
        const cv::Mat source = normalizeUint8(plane);
        profile.mark(N5Stage::UnitNormalize);

        const GuidedBaseResult decomposition =
            guidedBase(source, 7);
        profile.mark(N5Stage::UnitGuided7);

        const cv::Mat baseUint8 =
            quantizeUint8(decomposition.image);
        profile.mark(N5Stage::BaseQuantize);

        cv::Mat mappedUint8;
        clahe_->apply(baseUint8, mappedUint8);
        profile.mark(N5Stage::Clahe);

        const cv::Mat mapped = normalizeUint8(mappedUint8);
        cv::Mat reconstructed(
            plane.rows, plane.cols, CV_64FC1
        );

        for (int row = 0; row < plane.rows; ++row) {
            const auto* input = source.ptr<double>(row);
            const auto* base =
                decomposition.image.ptr<double>(row);
            const auto* mappedBase = mapped.ptr<double>(row);
            auto* output = reconstructed.ptr<double>(row);

            for (int column = 0; column < plane.cols; ++column) {
                const double residual =
                    input[column] - base[column];

                const double originalUnit =
    mappedBase[column] + residual;

                output[column] =
                    originalUnit
                    + correction.at<double>(row, column);
            }
        }

        // Required intermediate UNIT uint8 boundary.
        unitUint8 = quantizeUint8(reconstructed);
    }
    profile.mark(N5Stage::UnitReconstructQuantize);

    const cv::Mat unit = normalizeUint8(unitUint8);
    profile.mark(N5Stage::PostNormalize);
    const GuidedBaseResult small = guidedBase(unit, 3);
    profile.mark(N5Stage::PostGuided3);

    cv::Mat reconstructed(
        plane.rows, plane.cols, CV_64FC1
    );

    for (int row = 0; row < plane.rows; ++row) {
        const auto* input = unit.ptr<double>(row);
        const auto* base = small.image.ptr<double>(row);
        auto* output = reconstructed.ptr<double>(row);

        for (int column = 0; column < plane.cols; ++column) {
            const double residual =
                input[column] - base[column];

            const double attenuated = 0.5 * residual;
            output[column] = input[column] - attenuated;
        }
    }

    // Quantization also checks reconstruction finiteness.
    cv::Mat output = quantizeUint8(reconstructed);

    if (constant) {
        output = plane.clone();
    }
    profile.mark(N5Stage::PostReconstructQuantize);

    return {
        std::move(output),
        small.regularization,
        constant
    };
}

} // namespace qart::core::enhancement