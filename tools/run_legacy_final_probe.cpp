//
// Created by hakgu on 10/17/2026.
//

#include "te_rgf_rgb.hpp"
#include "core/enhancement/n5_thermal_enhancer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

cv::Mat readPlane(const fs::path& path)
{
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty() || image.type() != CV_8UC1) {
        throw std::runtime_error("Invalid grayscale PNG: " + path.string());
    }
    return image;
}

void save(const fs::path& path, const cv::Mat& image)
{
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write: " + path.string());
    }

    const cv::Mat decoded = readPlane(path);
    if (decoded.size() != image.size()
        || cv::norm(decoded, image, cv::NORM_INF) != 0.0) {
        throw std::runtime_error("PNG round-trip mismatch");
    }
}

cv::Mat toUint8(const cv::Mat& image)
{
    cv::Mat clipped, output;
    cv::min(cv::max(image, 0.0f), 1.0f, clipped);
    clipped.convertTo(output, CV_8U, 255.0);
    return output;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: run_legacy_final_probe INPUT_DIR NEW_OUTPUT_DIR\n";
        return 2;
    }

    try {
        const fs::path inputDirectory = argv[1];
        const fs::path outputDirectory = argv[2];

        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(inputDirectory)) {
            if (entry.is_regular_file()
                && entry.path().extension() == ".png") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());

        if (files.size() != 56) {
            throw std::runtime_error("Expected exactly 56 probe inputs");
        }

        if (!fs::create_directory(outputDirectory)) {
            throw std::runtime_error("Output directory already exists");
        }

        for (const char* mode :
             {"bypass", "n5", "legacy05", "legacy10"}) {
            if (!fs::create_directory(outputDirectory / mode)) {
                throw std::runtime_error("Cannot create mode directory");
            }
        }

        qart::core::enhancement::N5ThermalEnhancer enhancer;

        for (const auto& path : files) {
            const cv::Mat input = readPlane(path);
            const cv::Mat original = input.clone();

            const auto n5 = enhancer.enhance(input);
            if (n5.outputImage.type() != CV_8UC1
                || n5.outputImage.size() != input.size()) {
                throw std::runtime_error("N5 output contract violation");
            }

            cv::Mat dummyRgb;
            cv::cvtColor(input, dummyRgb, cv::COLOR_GRAY2BGR);
            const auto legacy = te::rgb_fusion(
                dummyRgb, input, nullptr, te::fusionParameters{});

            const auto& stages = legacy.dictionary;
            const cv::Mat& base = stages.at("base");
            const cv::Mat& fine = stages.at("D_fine_enh");
            const cv::Mat& structural = stages.at("D_struct_enh");

            cv::Mat base8;
            const cv::Mat scaled = base * 255.0f;
            scaled.convertTo(base8, CV_8U);

            // Explicit instance: do not reuse the legacy cached CLAHE helper.
            auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            cv::Mat mapped8, mapped;
            clahe->apply(base8, mapped8);
            mapped8.convertTo(mapped, CV_32F, 1.0 / 255.0);

            const cv::Mat legacy05 = toUint8(te::build_luminance(
                fine, structural, mapped,
                0.5f, 1.0f, 1.0f, 0.07f));

            const cv::Mat legacy10 = toUint8(te::build_luminance(
                fine, structural, mapped,
                1.0f, 1.0f, 1.0f, 0.07f));

            if (cv::norm(input, original, cv::NORM_INF) != 0.0) {
                throw std::runtime_error("Source was modified");
            }

            save(outputDirectory / "bypass" / path.filename(), original);
            save(outputDirectory / "n5" / path.filename(), n5.outputImage);
            save(outputDirectory / "legacy05" / path.filename(), legacy05);
            save(outputDirectory / "legacy10" / path.filename(), legacy10);

            std::cout << path.filename().string() << ": complete\n";
        }

        std::ofstream marker(outputDirectory / "COMPLETE.txt");
        marker << "inputs=56\noutputs=224\n"
               << "opencv=" << CV_VERSION << '\n';
        marker.close();
        if (!marker) {
            throw std::runtime_error("Cannot finalize completion marker");
        }

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << "\nTreat this run as incomplete.\n";
        return 1;
    }
}