//
// Created by hakgu on 10/17/2026.
//

#include "te_rgf_rgb.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Case {
    int number;
    const char* id;
};

const std::array<Case, 6> Cases{{
    {1, "m3fd_all_01415"},
    {2, "m3fd_all_00857"},
    {13, "tno_all_009"},
    {17, "llvip_train_010209"},
    {28, "msrs_train_01272N"},
    {39, "roadscene_all_FLIR_04593"}
}};

cv::Mat readPlane(const fs::path& path)
{
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty() || image.type() != CV_8UC1) {
        throw std::runtime_error(
            "Expected uint8 grayscale PNG: " + path.string());
    }
    return image;
}

cv::Mat unitPreview(const cv::Mat& image)
{
    cv::Mat clipped, output;
    cv::min(cv::max(image, 0.0f), 1.0f, clipped);
    clipped.convertTo(output, CV_8U, 255.0);
    return output;
}

void save(const fs::path& path, const cv::Mat& image)
{
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write: " + path.string());
    }
}

void statistics(
    std::ostream& csv, const char* id,
    const std::string& name, const cv::Mat& matrix)
{
    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(matrix, &minimum, &maximum);

    cv::Scalar mean, deviation;
    cv::meanStdDev(matrix, mean, deviation);

    const double pixels = static_cast<double>(matrix.total());

    csv << id << ',' << name << ','
        << minimum << ',' << maximum << ','
        << mean[0] << ',' << deviation[0] << ','
        << cv::countNonZero(matrix < 0.0f) / pixels << ','
        << cv::countNonZero(matrix > 1.0f) / pixels << '\n';

    if (!csv) {
        throw std::runtime_error("Cannot write stage statistics");
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr
            << "Usage: export_legacy_stages COHORT_RUN_DIR NEW_OUTPUT_DIR\n";
        return 2;
    }

    try {
        const fs::path cohort = argv[1];
        const fs::path output = argv[2];

        if (!fs::create_directory(output)) {
            throw std::runtime_error(
                "Output directory exists; choose a new directory");
        }

        std::ofstream csv(output / "stage_statistics.csv");
        if (!csv) {
            throw std::runtime_error("Cannot create stage_statistics.csv");
        }

        csv << std::setprecision(17)
            << "case_id,stage,min,max,mean,stddev,"
               "fraction_below_zero,fraction_above_one\n";

        for (const auto& selected : Cases) {
            const std::string number = std::to_string(selected.number);
            const cv::Mat input = readPlane(
                cohort / "fixture_planes"
                / (std::string(selected.id) + "__input.png"));

            const cv::Mat original = input.clone();
            const cv::Mat previous = readPlane(
                cohort / "comparison" / (number + "__legacy.png"));

            cv::Mat dummyRgb;
            cv::cvtColor(input, dummyRgb, cv::COLOR_GRAY2BGR);

            const te::fusionParameters parameters;
            const auto result = te::rgb_fusion(
                dummyRgb, input, nullptr, parameters);

            const auto& matrices = result.dictionary;
            const cv::Mat finalImage = unitPreview(matrices.at("L_T"));

            if (finalImage.size() != previous.size()
                || cv::norm(finalImage, previous, cv::NORM_INF) != 0.0
                || cv::norm(input, original, cv::NORM_INF) != 0.0) {
                throw std::runtime_error(
                    "Baseline/source mismatch: " + std::string(selected.id));
            }

            // Reproduce build_luminance() before its final clamp.
            cv::Mat fine;
            matrices.at("D_fine_enh").copyTo(fine);
            cv::min(cv::max(fine, -0.07f), 0.07f, fine);

            const float betaFine = 0.5f * parameters.b_fine;
            const float betaStructural = parameters.b_struct;
            const float betaBase = parameters.b_base;

            const cv::Mat preClamp =
                betaBase * matrices.at("B_enh")
                + betaFine * fine
                + betaStructural * matrices.at("D_struct_enh");

            cv::Mat clamped;
            cv::min(cv::max(preClamp, 0.0f), 1.0f, clamped);

            if (cv::norm(
                    clamped, matrices.at("L_T"), cv::NORM_INF) != 0.0) {
                throw std::runtime_error(
                    "Reconstruction mismatch: " + std::string(selected.id));
            }

            const fs::path caseDirectory = output / selected.id;
            if (!fs::create_directory(caseDirectory)) {
                throw std::runtime_error("Cannot create case directory");
            }

            cv::FileStorage raw(
                (caseDirectory / "stages.yml.gz").string(),
                cv::FileStorage::WRITE);

            if (!raw.isOpened()) {
                throw std::runtime_error("Cannot create raw stage archive");
            }

            for (const auto& entry : matrices) {
                if (!cv::checkRange(entry.second)) {
                    throw std::runtime_error(
                        "Non-finite matrix: " + entry.first);
                }
                raw << entry.first << entry.second;
                statistics(csv, selected.id, entry.first, entry.second);
            }

            const cv::Mat fineContribution = betaFine * fine;
            const cv::Mat structuralContribution =
                betaStructural * matrices.at("D_struct_enh");

            raw << "fine_contribution" << fineContribution;
            raw << "structural_contribution" << structuralContribution;
            raw << "pre_clamp" << preClamp;
            raw.release();

            statistics(csv, selected.id, "fine_contribution",
                       fineContribution);
            statistics(csv, selected.id, "structural_contribution",
                       structuralContribution);
            statistics(csv, selected.id, "pre_clamp", preClamp);

            save(caseDirectory / "input.png", input);
            save(caseDirectory / "base.png",
                 unitPreview(matrices.at("base")));
            save(caseDirectory / "base_clahe.png",
                 unitPreview(matrices.at("B_enh")));
            save(caseDirectory / "final.png", finalImage);

            // Fixed signed display range [-0.15,+0.15].
            // Neutral zero is middle gray. Raw values remain in stages.yml.gz.
            const cv::Mat fineDisplay =
                0.5f + fineContribution / 0.30f;
            const cv::Mat structuralDisplay =
                0.5f + structuralContribution / 0.30f;

            save(caseDirectory / "fine_signed_preview.png",
                 unitPreview(fineDisplay));
            save(caseDirectory / "structural_signed_preview.png",
                 unitPreview(structuralDisplay));

            cv::Mat below, above, clipping;
            cv::compare(preClamp, 0.0f, below, cv::CMP_LT);
            cv::compare(preClamp, 1.0f, above, cv::CMP_GT);
            cv::bitwise_or(below, above, clipping);
            save(caseDirectory / "pre_clamp_outside_range.png", clipping);

            std::cout << selected.id
                      << ": exact baseline match; stages exported\n";
        }

        csv.close();
        if (!csv) {
            throw std::runtime_error("Cannot finalize stage statistics");
        }

        std::ofstream complete(output / "COMPLETE.txt");
        complete
            << "cases=6\n"
            << "configuration=legacy_run_triage_defaults\n"
            << "baseline_pixel_matches=6\n"
            << "reconstruction_float_matches=6\n"
            << "signed_preview_range=-0.15,+0.15\n"
            << "raw_storage=OpenCV_FileStorage_float_matrices\n"
            << "opencv=" << CV_VERSION << '\n'
            << "cohort_directory=" << fs::absolute(cohort).string() << '\n';

        complete.close();
        if (!complete) {
            throw std::runtime_error("Cannot finalize completion marker");
        }

        std::cout << "Complete: 6 cases; no algorithm changes\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what()
                  << "\nTreat this export as incomplete.\n";
        return 1;
    }
}