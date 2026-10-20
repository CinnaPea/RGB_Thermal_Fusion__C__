#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "n5_thermal_enhancer.h"
#include "te_rgf_rgb.hpp"
#include "legacy_thermal_candidate.h"

namespace qart::experiment {

qart::core::enhancement::N5EnhancementResult enhance(
    const cv::Mat& plane,
    const cv::Mat& correction);

}

namespace {

namespace fs = std::filesystem;

cv::Mat readPlane(const fs::path& path)
{
    cv::Mat image = cv::imread(
        path.string(),
        cv::IMREAD_UNCHANGED
    );

    if (
        image.empty()
        || image.dims != 2
        || image.type() != CV_8UC1
    ) {
        throw std::runtime_error(
            "Expected decoded CV_8UC1 PNG: "
            + path.string()
        );
    }

    return image;
}

void requireEqual(
    const cv::Mat& actual,
    const cv::Mat& expected,
    const std::string& context)
{
    if (
        actual.size() != expected.size()
        || actual.type() != expected.type()
        || cv::norm(
            actual,
            expected,
            cv::NORM_INF
        ) != 0.0
    ) {
        throw std::runtime_error(
            "Pixel equivalence failed: " + context
        );
    }
}

void writeVerified(
    const fs::path& path,
    const cv::Mat& image)
{
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error(
            "Cannot write: " + path.string()
        );
    }

    requireEqual(
        readPlane(path),
        image,
        "PNG round trip: " + path.string()
    );
}

cv::Mat detailCorrection(
    const cv::Mat& input,
    const cv::Mat& legacyBase,
    const cv::Mat& treatedFine,
    const cv::Mat& treatedStructure)
{
    cv::Mat source;
    input.convertTo(
        source,
        CV_32F,
        1.0 / 255.0
    );

    // Preserve the legacy float32 calculation boundary.
    const cv::Mat originalResidual =
        source - legacyBase;

    const cv::Mat treatedResidual =
        treatedFine + treatedStructure;

    const cv::Mat correction32 =
        treatedResidual - originalResidual;

    cv::Mat correction64;
    correction32.convertTo(
        correction64,
        CV_64F
    );

    return correction64;
}

void writeMetrics(
    std::ostream& report,
    const std::string& caseId,
    const std::string& mode,
    const cv::Mat& image)
{
    cv::Scalar mean;
    cv::Scalar deviation;
    cv::meanStdDev(image, mean, deviation);

    cv::Mat laplacian;
    cv::Laplacian(
        image,
        laplacian,
        CV_64F,
        1,
        1.0,
        0.0,
        cv::BORDER_REFLECT_101
    );

    cv::Scalar laplacianMean;
    cv::Scalar laplacianDeviation;
    cv::meanStdDev(
        laplacian,
        laplacianMean,
        laplacianDeviation
    );

    cv::Mat gradientX;
    cv::Mat gradientY;
    cv::Mat magnitude;

    cv::Sobel(
        image,
        gradientX,
        CV_64F,
        1, 0, 3,
        1.0, 0.0,
        cv::BORDER_REFLECT_101
    );

    cv::Sobel(
        image,
        gradientY,
        CV_64F,
        0, 1, 3,
        1.0, 0.0,
        cv::BORDER_REFLECT_101
    );

    cv::magnitude(
        gradientX,
        gradientY,
        magnitude
    );

    const double pixelCount =
        static_cast<double>(image.total());

    report
        << caseId << ','
        << mode << ','
        << deviation[0] << ','
        << laplacianDeviation[0]
            * laplacianDeviation[0] << ','
        << cv::mean(magnitude)[0] << ','
        << cv::countNonZero(image == 0)
            / pixelCount << ','
        << cv::countNonZero(image == 255)
            / pixelCount << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const bool probeMode =
            argc == 5
            && std::string(argv[4]) == "--probe";

        if (argc != 4 && !probeMode) {
            throw std::invalid_argument(
                "Usage: evaluate_selective_n5 "
                "OUTPUT_DIR INPUT_DIR "
                "LEGACY_REFERENCE_DIR [--probe]"
            );
        }

        const fs::path outputDirectory =
            fs::absolute(argv[1]);

        const fs::path inputDirectory =
            fs::absolute(argv[2]);

        // In probe mode argv[3] is "-", and is unused.
        const fs::path legacyReferenceDirectory =
            probeMode
                ? fs::path{}
                : fs::absolute(argv[3]);

        if (!fs::is_directory(inputDirectory)) {
            throw std::runtime_error(
                "Input directory does not exist: "
                + inputDirectory.string()
            );
        }

        if (
            !probeMode
            && !fs::is_directory(
                legacyReferenceDirectory
            )
        ) {
            throw std::runtime_error(
                "Legacy reference directory does not exist: "
                + legacyReferenceDirectory.string()
            );
        }

        if (fs::exists(outputDirectory)) {
            throw std::runtime_error(
                "Output directory already exists; "
                "choose a fresh run directory"
            );
        }

        const std::string suffix =
            probeMode ? ".png" : "__input.png";

        const std::size_t expectedCases =
            probeMode ? 56 : 40;

        std::vector<fs::path> inputs;

        for (
            const auto& entry :
            fs::directory_iterator(inputDirectory)
        ) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const std::string name =
                entry.path().filename().string();

            if (
                name.size() > suffix.size()
                && name.compare(
                    name.size() - suffix.size(),
                    suffix.size(),
                    suffix
                ) == 0
            ) {
                inputs.push_back(entry.path());
            }
        }

        std::sort(inputs.begin(), inputs.end());

        if (inputs.size() != expectedCases) {
            throw std::runtime_error(
                "Expected "
                + std::to_string(expectedCases)
                + " inputs; found "
                + std::to_string(inputs.size())
            );
        }

        fs::create_directories(outputDirectory);

        std::ofstream metrics(
            outputDirectory / "metrics.csv"
        );

        std::ofstream inputReport(
            outputDirectory / "inputs.tsv"
        );

        if (!metrics || !inputReport) {
            throw std::runtime_error(
                "Cannot open output reports"
            );
        }

        metrics << std::setprecision(17);

        metrics
            << "case_id,mode,rms_contrast,"
            << "laplacian_variance,sobel_mean,"
            << "min_clip_fraction,max_clip_fraction\n";

        inputReport << "case_id\tinput_path\n";

        cv::setNumThreads(8);

        qart::core::enhancement::N5ThermalEnhancer
            acceptedN5;
        qart::experiment::LegacyThermalCandidate extractedLegacy;

        const std::vector<te::Scales_RGF> scales{
            {1, 4, 1e-3f, 1.2f},
            {1, 4, 1e-3f, 2.0f},
            {1, 4, 1e-3f, 3.0f}
        };

        std::size_t metricRows = 0;
        std::size_t completedCases = 0;

        for (const auto& inputPath : inputs) {
            const std::string filename =
                inputPath.filename().string();

            const std::string caseId =
                filename.substr(
                    0,
                    filename.size() - suffix.size()
                );

            if (
                caseId.find_first_of(",\t\r\n")
                != std::string::npos
            ) {
                throw std::runtime_error(
                    "Invalid report case ID"
                );
            }

            const cv::Mat input =
                readPlane(inputPath);

            const cv::Mat original =
                input.clone();

            const cv::Mat n5 =
                acceptedN5.enhance(input).outputImage;

            if (!probeMode) {
                const cv::Mat expectedN5 =
                    readPlane(
                        inputDirectory
                        / (caseId + "__expected_n5.png")
                    );

                requireEqual(
                    n5,
                    expectedN5,
                    caseId + ": accepted N5 reference"
                );
            }

            // Verify the copied N5 calculation in both modes.
            const cv::Mat zeroCorrection =
                cv::Mat::zeros(
                    input.size(),
                    CV_64FC1
                );

            const cv::Mat identity =
                qart::experiment::enhance(
                    input,
                    zeroCorrection
                ).outputImage;

            requireEqual(
                identity,
                n5,
                caseId + ": zero-correction identity"
            );

            cv::Mat d1;
            cv::Mat d2;
            cv::Mat d3;
            cv::Mat legacyBase;

            te::rgf_decompose(
                input,
                scales,
                d1, d2, d3,
                legacyBase
            );

            cv::Mat fineWeights;
            cv::Mat structuralWeights;

            te::weight_map(
                input,
                7,
                0.3f, 1.0f,
                0.8f, 1.4f,
                0.5f,
                fineWeights,
                structuralWeights
            );

            cv::GaussianBlur(
                fineWeights,
                fineWeights,
                cv::Size(5, 5),
                1.0, 1.0,
                cv::BORDER_REFLECT_101
            );

            cv::GaussianBlur(
                structuralWeights,
                structuralWeights,
                cv::Size(5, 5),
                1.0, 1.0,
                cv::BORDER_REFLECT_101
            );

            cv::Mat treatedFine;
            cv::Mat treatedStructure;

            te::enhance_details(
                d1, d2, d3,
                fineWeights,
                structuralWeights,
                treatedFine,
                treatedStructure,
                0.07f
            );

            // Control: clipping without adaptive weights.
            const cv::Mat ones(
                input.size(),
                CV_32FC1,
                cv::Scalar(1.0)
            );

            cv::Mat clippedFine;
            cv::Mat unweightedStructure;

            te::enhance_details(
                d1, d2, d3,
                ones,
                ones,
                clippedFine,
                unweightedStructure,
                0.07f
            );

            // Standalone legacy comparator: grid8, fine1.
            // Avoid the legacy helper's cached CLAHE state.
            const cv::Mat scaledBase =
                legacyBase * 255.0f;

            cv::Mat legacyBase8;
            scaledBase.convertTo(
                legacyBase8,
                CV_8U
            );

            auto legacyClahe = cv::createCLAHE(
                2.0,
                cv::Size(8, 8)
            );

            cv::Mat mappedLegacy8;
            legacyClahe->apply(
                legacyBase8,
                mappedLegacy8
            );

            cv::Mat mappedLegacy;
            mappedLegacy8.convertTo(
                mappedLegacy,
                CV_32F,
                1.0 / 255.0
            );

            const cv::Mat legacyFloat =
                te::build_luminance(
                    treatedFine,
                    treatedStructure,
                    mappedLegacy,
                    1.0f,
                    1.0f,
                    1.0f,
                    0.07f
                );

            cv::Mat legacy;
            legacyFloat.convertTo(
                legacy,
                CV_8U,
                255.0
            );

            const cv::Mat extracted = extractedLegacy.enhance(input);

            requireEqual(
                extracted,
                legacy,
                caseId + ": standalone legacy extraction"
            );

            if (!probeMode) {
                const cv::Mat expectedLegacy =
                    readPlane(
                        legacyReferenceDirectory
                        / caseId
                        / "legacy_grid8_fine10.png"
                    );

                requireEqual(
                    legacy,
                    expectedLegacy,
                    caseId + ": legacy parent reference"
                );
            }

            const cv::Mat clipCorrection =
                detailCorrection(
                    input,
                    legacyBase,
                    clippedFine,
                    unweightedStructure
                );

            const cv::Mat weightedCorrection =
                detailCorrection(
                    input,
                    legacyBase,
                    treatedFine,
                    treatedStructure
                );

            const cv::Mat clipCandidate =
                qart::experiment::enhance(
                    input,
                    clipCorrection
                ).outputImage;

            const cv::Mat weightedCandidate =
                qart::experiment::enhance(
                    input,
                    weightedCorrection
                ).outputImage;

            const std::vector<
                std::pair<std::string, cv::Mat>
            > results{
                {"bypass", input},
                {"n5", n5},
                {"legacy_grid8_fine10", legacy},
                {"n5_legacy_clip_control", clipCandidate},
                {"n5_legacy_weighted", weightedCandidate}
            };

            const fs::path caseDirectory =
                outputDirectory / caseId;

            fs::create_directory(caseDirectory);

            for (const auto& result : results) {
                writeVerified(
                    caseDirectory
                    / (result.first + ".png"),
                    result.second
                );

                writeMetrics(
                    metrics,
                    caseId,
                    result.first,
                    result.second
                );

                ++metricRows;
            }

            requireEqual(
                input,
                original,
                caseId + ": source unchanged"
            );

            inputReport
                << caseId << '\t'
                << inputPath.generic_string()
                << '\n';

            ++completedCases;

            std::cout
                << caseId
                << ": N5 identity PASS; "
                << "5 outputs verified\n";
        }

        metrics.close();
        inputReport.close();

        if (metrics.fail() || inputReport.fail()) {
            throw std::runtime_error(
                "Report write failed"
            );
        }

        if (
            completedCases != expectedCases
            || metricRows != expectedCases * 5
        ) {
            throw std::runtime_error(
                "Unexpected final report counts"
            );
        }

        std::ofstream complete(
            outputDirectory / "COMPLETE.txt"
        );

        complete
            << "test_mode="
            << (probeMode ? "fixed_probe" : "native40")
            << '\n'
            << "cases=" << completedCases << '\n'
            << "outputs=" << metricRows << '\n'
            << "metric_rows=" << metricRows << '\n'
            << "n5_zero_correction_exact="
            << completedCases << '/'
            << completedCases << '\n'
            << "legacy_extraction_exact="
            << completedCases << '/'
            << completedCases << '\n';

        if (probeMode) {
            complete
                << "parent_reference_checks="
                << "native40_run_only\n";
        }
        else {
            complete
                << "n5_parent_exact="
                << completedCases << '/'
                << completedCases << '\n'
                << "legacy_parent_exact="
                << completedCases << '/'
                << completedCases << '\n';
        }

        complete
            << "OpenCV=" << CV_VERSION << '\n'
            << "threads=" << cv::getNumThreads() << '\n'
            << "status=experimental_not_accepted\n";

        complete.close();

        if (complete.fail()) {
            throw std::runtime_error(
                "Cannot write completion marker"
            );
        }

        std::cout
            << "Complete: "
            << completedCases
            << " cases; "
            << metricRows
            << " report rows\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Evaluation failed: "
            << error.what()
            << "\nTreat this run as incomplete.\n";

        return 1;
    }
}