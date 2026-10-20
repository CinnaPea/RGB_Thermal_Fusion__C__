//
// Created by hakgu on 10/17/2026.
//

#include "te_rgf_rgb.hpp"
#include "triage_metrics.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

cv::Mat thermalOnly(const cv::Mat& input)
{
    const te::fusionParameters p;

    // Exact default configuration from rgb_fusion().
    const std::vector<te::Scales_RGF> scales{
        {1, 4, 1e-3f, 1.2f},
        {1, 4, 1e-3f, 2.0f},
        {1, 4, 1e-3f, 3.0f}
    };

    const cv::Mat thermal8 =
        te::percentile8bit(input, p.percentile_lo, p.percentile_hi);

    cv::Mat d1, d2, d3, base;
    te::rgf_decompose(thermal8, scales, d1, d2, d3, base);

    cv::Mat fineWeight, structuralWeight;
    te::weight_map(
        thermal8, p.ksize_H,
        0.3f, 1.0f, 0.8f, 1.4f, 0.5f,
        fineWeight, structuralWeight);

    cv::GaussianBlur(
        fineWeight, fineWeight, cv::Size(5, 5),
        1.0, 1.0, cv::BORDER_REFLECT101);

    cv::GaussianBlur(
        structuralWeight, structuralWeight, cv::Size(5, 5),
        1.0, 1.0, cv::BORDER_REFLECT101);

    cv::Mat fine, structural;
    te::enhance_details(
        d1, d2, d3, fineWeight, structuralWeight,
        fine, structural, 0.07f);

    const cv::Mat enhancedBase =
        te::clahe(base, 2.0, cv::Size(64, 64));

    return te::build_luminance(
        fine, structural, enhancedBase,
        0.5f * p.b_fine, p.b_struct, p.b_base, 0.07f);
}

cv::Mat toUint8(const cv::Mat& image)
{
    cv::Mat clipped, result;
    cv::min(cv::max(image, 0.0f), 1.0f, clipped);
    clipped.convertTo(result, CV_8U, 255.0);
    return result;
}

void saveImage(const fs::path& path, const cv::Mat& image)
{
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write: " + path.string());
    }
}

void writeMetrics(
    std::ostream& csv, int caseNumber,
    const char* mode, const cv::Mat& image)
{
    const auto m = triage::metrics(image);
    const double pixels = static_cast<double>(image.total());

    csv << caseNumber << ',' << mode << ','
        << image.cols << ',' << image.rows << ','
        << m.entropy << ',' << m.rms << ','
        << m.lap_var << ',' << m.sobel << ','
        << cv::countNonZero(image == 0) / pixels << ','
        << cv::countNonZero(image == 255) / pixels << '\n';

    if (!csv) {
        throw std::runtime_error("Cannot write metrics CSV");
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr
            << "Usage: evaluate_legacy_thermal OUTPUT_DIR "
               "INPUT_IMAGE [INPUT_IMAGE ...]\n";
        return 2;
    }

    try {
        const fs::path outputDirectory = argv[1];

        // A fresh directory keeps partial or previous runs distinguishable.
        if (!fs::create_directory(outputDirectory)) {
            throw std::runtime_error(
                "Output directory already exists; use a fresh directory");
        }

        std::ofstream csv(outputDirectory / "metrics.csv");
        std::ofstream manifest(outputDirectory / "inputs.txt");

        if (!csv || !manifest) {
            throw std::runtime_error("Cannot create report files");
        }

        csv << std::setprecision(17);
        csv << "case_number,mode,width,height,entropy,rms,"
               "laplacian_variance,sobel_energy,"
               "minimum_clip_fraction,maximum_clip_fraction\n";

        for (int argument = 2; argument < argc; ++argument) {
            const int caseNumber = argument - 1;
            const fs::path inputPath = argv[argument];

            // First comparison uses the accepted uint8 thermal-plane domain.
            const cv::Mat decoded = cv::imread(
    inputPath.string(), cv::IMREAD_UNCHANGED);

            if (decoded.empty()) {
                throw std::runtime_error(
                    "Cannot decode image: " + inputPath.string());
            }

            if (decoded.depth() != CV_8U) {
                throw std::runtime_error(
                    "This comparison requires uint8 input: "
                    + inputPath.string());
            }

            cv::Mat input;
            switch (decoded.channels()) {
                case 1:
                    input = decoded.clone();
                    break;
                case 3:
                    cv::cvtColor(decoded, input, cv::COLOR_BGR2GRAY);
                    break;
                case 4:
                    cv::cvtColor(decoded, input, cv::COLOR_BGRA2GRAY);
                    break;
                default:
                    throw std::runtime_error(
                        "Unsupported channel count: "
                        + std::to_string(decoded.channels()));
            }

            std::cout << inputPath.filename().string()
                      << ": decoded channels=" << decoded.channels()
                      << "; grayscale=" << input.cols << 'x' << input.rows
                      << '\n';

            const cv::Mat original = input.clone();
            const cv::Mat extracted = thermalOnly(input);

            // Reference-path check only: synthetic RGB does not contribute
            // to L_T, which is constructed before luminance fusion.
            cv::Mat dummyRgb;
            cv::cvtColor(input, dummyRgb, cv::COLOR_GRAY2BGR);

            const auto reference =
                te::rgb_fusion(dummyRgb, input, nullptr,
                               te::fusionParameters{});

            const cv::Mat& referenceThermal =
                reference.dictionary.at("L_T");

            if (!cv::checkRange(extracted)
                || extracted.type() != referenceThermal.type()
                || extracted.size() != referenceThermal.size()
                || cv::norm(extracted, referenceThermal,
                            cv::NORM_INF) != 0.0) {
                throw std::runtime_error(
                    "Thermal extraction differs from repository L_T");
            }

            if (cv::norm(input, original, cv::NORM_INF) != 0.0) {
                throw std::runtime_error("Source image was modified");
            }

            const cv::Mat hybrid = toUint8(extracted);
            const std::string prefix = std::to_string(caseNumber);

            saveImage(outputDirectory / (prefix + "__input.png"), input);
            saveImage(outputDirectory / (prefix + "__legacy.png"), hybrid);

            writeMetrics(csv, caseNumber, "bypass", input);
            writeMetrics(csv, caseNumber, "legacy_default", hybrid);

            manifest << caseNumber << '\t'
                     << fs::absolute(inputPath).string() << '\n';

            if (!manifest) {
                throw std::runtime_error("Cannot write input manifest");
            }

            std::cout << caseNumber
                      << ": exact L_T match; source unchanged\n";
        }

        csv.close();
        manifest.close();
        if (!csv || !manifest) {
            throw std::runtime_error("Cannot finalize report files");
        }

        std::ofstream complete(outputDirectory / "COMPLETE.txt");
        complete << "source_commit=edf446d70cbdacc53504576ca548afd61824b084\n"
                 << "configuration=run_triage_defaults\n"
                 << "opencv=" << CV_VERSION << '\n'
                 << "cases=" << argc - 2 << '\n';
        complete.close();
        if (!complete) {
            throw std::runtime_error("Cannot finalize completion marker");
        }

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Evaluation failed: " << error.what()
                  << "\nTreat this run as incomplete.\n";
        return 1;
    }
}