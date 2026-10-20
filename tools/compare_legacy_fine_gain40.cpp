//
// Created by hakgu on 10/17/2026.
//

#include "te_rgf_rgb.hpp"
#include "triage_metrics.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

    struct Case {
        int number;
        std::string id;
    };

    std::vector<Case> readCases(const fs::path& path)
    {
        std::ifstream stream(path);
        if (!stream) {
            throw std::runtime_error("Cannot open case list: " + path.string());
        }

        std::vector<Case> cases;
        std::unordered_set<std::string> seen;
        std::string line;

        while (std::getline(stream, line)) {
            if (line.empty()) {
                throw std::runtime_error("Empty case-list entry");
            }

            std::istringstream entry(line);
            Case selected{};
            std::string trailing;

            if (!(entry >> selected.number >> selected.id)
                || (entry >> trailing)) {
                throw std::runtime_error("Malformed case-list entry");
                }

            const bool safeId = !selected.id.empty()
                && std::all_of(
                    selected.id.begin(), selected.id.end(),
                    [](unsigned char character) {
                        return (character >= 'a' && character <= 'z')
                            || (character >= 'A' && character <= 'Z')
                            || (character >= '0' && character <= '9')
                            || character == '_';
                    });

            if (!safeId
                || selected.number != static_cast<int>(cases.size()) + 1
                || !seen.insert(selected.id).second) {
                throw std::runtime_error(
                    "Invalid, duplicate, or unordered case-list entry");
                }

            cases.push_back(selected);
        }

        if (!stream.eof() || cases.size() != 40) {
            throw std::runtime_error("Expected exactly 40 case-list entries");
        }

        return cases;
    }

cv::Mat readPng(const fs::path& path)
{
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty() || image.type() != CV_8UC1) {
        throw std::runtime_error("Invalid grayscale PNG: " + path.string());
    }
    return image;
}

cv::Mat readFloat(cv::FileStorage& storage, const char* name)
{
    cv::Mat matrix;
    storage[name] >> matrix;

    if (matrix.empty() || matrix.type() != CV_32FC1
        || !cv::checkRange(matrix)) {
        throw std::runtime_error(
            "Invalid floating-point stage: " + std::string(name));
    }
    return matrix;
}

cv::Mat toUint8(const cv::Mat& image)
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
    const cv::Mat decoded = readPng(path);
    if (decoded.size() != image.size()
        || cv::norm(decoded, image, cv::NORM_INF) != 0.0) {
        throw std::runtime_error("PNG round-trip mismatch");
    }
}

void metrics(
    std::ostream& csv, const std::string& id,
    const char* mode, const cv::Mat& image)
{
    const auto m = triage::metrics(image);
    const double pixels = static_cast<double>(image.total());

    csv << id << ',' << mode << ','
        << image.cols << ',' << image.rows << ','
        << m.entropy << ',' << m.rms << ','
        << m.lap_var << ',' << m.sobel << ','
        << cv::countNonZero(image == 0) / pixels << ','
        << cv::countNonZero(image == 255) / pixels << '\n';

    if (!csv) {
        throw std::runtime_error("Cannot write metrics");
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr
            << "Usage: compare_legacy_fine_gain40 "
               "CASE_LIST COHORT_RUN_DIR GRID8_BASELINE_DIR NEW_OUTPUT_DIR\n";
        return 2;
    }

    try {
        const fs::path caseList = argv[1];
        const fs::path cohort = argv[2];
        const fs::path gridBaseline = argv[3];
        const fs::path output = argv[4];
        const auto Cases = readCases(caseList);

        if (!fs::create_directory(output)) {
            throw std::runtime_error(
                "Output directory exists; choose a new directory");
        }

        std::ofstream csv(output / "metrics.csv");
        std::ofstream excursions(output / "reconstruction_ranges.csv");
        if (!csv || !excursions) {
            throw std::runtime_error("Cannot create reports");
        }

        csv << std::setprecision(17)
            << "case_id,mode,width,height,entropy,rms,"
               "laplacian_variance,sobel_energy,"
               "minimum_clip_fraction,maximum_clip_fraction\n";

        excursions << std::setprecision(17)
                   << "case_id,fine_gain,min,max,"
                      "fraction_below_zero,fraction_above_one\n";

        for (const auto& selected : Cases) {
            const cv::Mat input = readPng(
                        cohort / "fixture_planes" / (selected.id + "__input.png"));
            const cv::Mat original = input.clone();

            cv::Mat dummyRgb;
            cv::cvtColor(input, dummyRgb, cv::COLOR_GRAY2BGR);

            const auto legacy = te::rgb_fusion(
                dummyRgb, input, nullptr, te::fusionParameters{});

            const auto& dictionary = legacy.dictionary;
            const cv::Mat base = dictionary.at("base");
            const cv::Mat savedBase = dictionary.at("B_enh");
            const cv::Mat fine = dictionary.at("D_fine_enh");
            const cv::Mat structural = dictionary.at("D_struct_enh");
            const cv::Mat savedFinal = dictionary.at("L_T");

            for (const cv::Mat* matrix :
                 {&base, &savedBase, &fine, &structural, &savedFinal}) {
                if (matrix->empty()
                    || matrix->type() != CV_32FC1
                    || !cv::checkRange(*matrix)) {
                    throw std::runtime_error("Invalid legacy intermediate matrix");
                    }
                 }

            if (cv::norm(input, original, cv::NORM_INF) != 0.0) {
                throw std::runtime_error("Legacy modified its source");
            }

            for (const cv::Mat* matrix :
                 {&savedBase, &fine, &structural, &savedFinal}) {
                if (matrix->size() != base.size()) {
                    throw std::runtime_error("Stage geometry mismatch");
                }
            }

            const std::string number = std::to_string(selected.number);
            const fs::path comparisons = cohort / "comparison";

            const cv::Mat bypass =
                readPng(comparisons / (number + "__bypass.png"));
            const cv::Mat n5 =
                readPng(comparisons / (number + "__n5.png"));
            const cv::Mat previous =
                readPng(comparisons / (number + "__legacy.png"));

            for (const cv::Mat* image : {&bypass, &n5, &previous}) {
                if (image->size() != base.size()) {
                    throw std::runtime_error("Comparison geometry mismatch");
                }
            }

            if (bypass.size() != original.size()
                    || cv::norm(bypass, original, cv::NORM_INF) != 0.0) {
                        throw std::runtime_error("Fixture/cohort input mismatch");
            }

            const fs::path destination = output / selected.id;
            if (!fs::create_directory(destination)) {
                throw std::runtime_error("Cannot create case directory");
            }

            metrics(csv, selected.id, "bypass", bypass);
            metrics(csv, selected.id, "n5_accepted", n5);
            save(destination / "bypass.png", bypass);
            save(destination / "n5.png", n5);

            // Match the original base-to-uint8 conversion.
            cv::Mat base8;
            const cv::Mat scaled = base * 255.0f;
            scaled.convertTo(base8, CV_8U);

            for (const int grid : {64, 8}) {
                // Verify the historical grid64 baseline before testing anything new.
auto clahe64 = cv::createCLAHE(2.0, cv::Size(64, 64));
cv::Mat mapped64u8, mapped64;
clahe64->apply(base8, mapped64u8);
mapped64u8.convertTo(mapped64, CV_32F, 1.0 / 255.0);

const cv::Mat reconstructed64 = te::build_luminance(
    fine, structural, mapped64,
    0.5f, 1.0f, 1.0f, 0.07f);

if (cv::norm(mapped64, savedBase, cv::NORM_INF) != 0.0
    || cv::norm(reconstructed64, savedFinal, cv::NORM_INF) != 0.0
    || cv::norm(toUint8(reconstructed64), previous,
                cv::NORM_INF) != 0.0) {
    throw std::runtime_error(
        "Historical grid64 baseline mismatch: " + selected.id);
}

// Both fine-gain variants share exactly the same grid8 base.
auto clahe8 = cv::createCLAHE(2.0, cv::Size(8, 8));
cv::Mat mapped8u8, mapped8;
clahe8->apply(base8, mapped8u8);
mapped8u8.convertTo(mapped8, CV_32F, 1.0 / 255.0);

const cv::Mat savedGrid8 = readPng(
    gridBaseline / selected.id / "legacy_grid8.png");

if (savedGrid8.size() != base.size()) {
    throw std::runtime_error("Saved grid8 geometry mismatch");
}

cv::Mat clippedFine;
cv::min(cv::max(fine, -0.07f), 0.07f, clippedFine);

for (const float fineGain : {0.5f, 1.0f}) {
    const cv::Mat reconstructed = te::build_luminance(
        fine, structural, mapped8,
        fineGain, 1.0f, 1.0f, 0.07f);

    const cv::Mat result = toUint8(reconstructed);

    if (fineGain == 0.5f
        && cv::norm(result, savedGrid8, cv::NORM_INF) != 0.0) {
        throw std::runtime_error(
            "Accepted research grid8 baseline mismatch: " + selected.id);
    }

    // Mirror the original reconstruction order before its final clamp.
    const cv::Mat preClamp =
        1.0f * mapped8
        + fineGain * clippedFine
        + 1.0f * structural;

    cv::Mat checkedClamp;
    cv::min(cv::max(preClamp, 0.0f), 1.0f, checkedClamp);
    if (!cv::checkRange(preClamp)
        || cv::norm(checkedClamp, reconstructed, cv::NORM_INF) != 0.0) {
        throw std::runtime_error(
            "Pre-clamp reconstruction mismatch: " + selected.id);
    }

    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(preClamp, &minimum, &maximum);
    const double pixels = static_cast<double>(base.total());

    excursions << selected.id << ',' << fineGain << ','
               << minimum << ',' << maximum << ','
               << cv::countNonZero(preClamp < 0.0f) / pixels << ','
               << cv::countNonZero(preClamp > 1.0f) / pixels << '\n';

    if (!excursions) {
        throw std::runtime_error("Cannot write reconstruction ranges");
    }

    const std::string label = fineGain == 0.5f
        ? "legacy_grid8_fine05"
        : "legacy_grid8_fine10";

    metrics(csv, selected.id, label.c_str(), result);
    save(destination / (label + ".png"), result);
}
            }
            std::cout << selected.id
                      << ": grid64 and grid8 baselines exact; "
                         "fine gains 0.5/1.0 exported\n";
        }

        csv.close();
        excursions.close();
        if (!csv || !excursions) {
            throw std::runtime_error("Cannot finalize reports");
        }

        std::ofstream complete(output / "COMPLETE.txt");
        complete << "cases=" << Cases.size() << '\n'
         << "metric_rows=" << Cases.size() * 4 << '\n'
         << "grid64_baseline_exact_matches=" << Cases.size() << '\n'
         << "grid8_baseline_exact_matches=" << Cases.size() << '\n'
         << "changed_parameter=final_fine_coefficient_0.5_to_1.0\n"
         << "clahe_grid=8x8\n"
         << "clip_limit=2\n"
         << "fine_clip=0.07\n"
         << "structural_gain=1\n"
         << "base_gain=1\n"
         << "additional_postprocessing=none\n"
         << "opencv=" << CV_VERSION << '\n'
         << "case_list=" << fs::absolute(caseList).string() << '\n'
         << "cohort=" << fs::absolute(cohort).string() << '\n'
         << "grid8_baseline="
         << fs::absolute(gridBaseline).string() << '\n';
        complete.close();

        if (!complete) {
            throw std::runtime_error("Cannot finalize completion marker");
        }

        std::cout << "Complete: " << Cases.size()
                  << " cases; " << Cases.size() * 4
                  << " rows\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what()
                  << "\nTreat this experiment as incomplete.\n";
        return 1;
    }
}