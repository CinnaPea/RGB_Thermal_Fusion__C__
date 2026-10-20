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
    std::ostream& csv, const char* id,
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
    if (argc != 4) {
        std::cerr
            << "Usage: compare_legacy_clahe_grid "
               "STAGES_DIR COHORT_RUN_DIR NEW_OUTPUT_DIR\n";
        return 2;
    }

    try {
        const fs::path stages = argv[1];
        const fs::path cohort = argv[2];
        const fs::path output = argv[3];

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
                   << "case_id,grid,min,max,"
                      "fraction_below_zero,fraction_above_one\n";

        for (const auto& selected : Cases) {
            const fs::path source = stages / selected.id;

            cv::FileStorage raw(
                (source / "stages.yml.gz").string(),
                cv::FileStorage::READ);
            if (!raw.isOpened()) {
                throw std::runtime_error("Cannot open stage archive");
            }

            const cv::Mat base = readFloat(raw, "base");
            const cv::Mat savedBase = readFloat(raw, "B_enh");
            const cv::Mat fine = readFloat(raw, "D_fine_enh");
            const cv::Mat structural = readFloat(raw, "D_struct_enh");
            const cv::Mat savedFinal = readFloat(raw, "L_T");

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

            if (cv::norm(
                    bypass, readPng(source / "input.png"),
                    cv::NORM_INF) != 0.0) {
                throw std::runtime_error("Stage/cohort input mismatch");
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
                auto clahe = cv::createCLAHE(2.0, cv::Size(grid, grid));

                cv::Mat mapped8, mapped;
                clahe->apply(base8, mapped8);
                mapped8.convertTo(mapped, CV_32F, 1.0 / 255.0);

                const cv::Mat reconstructed = te::build_luminance(
                    fine, structural, mapped,
                    0.5f, 1.0f, 1.0f, 0.07f);
                const cv::Mat result = toUint8(reconstructed);

                if (grid == 64) {
                    if (cv::norm(mapped, savedBase, cv::NORM_INF) != 0.0
                        || cv::norm(
                            reconstructed, savedFinal, cv::NORM_INF) != 0.0
                        || cv::norm(result, previous, cv::NORM_INF) != 0.0) {
                        throw std::runtime_error(
                            "64x64 baseline mismatch: "
                            + std::string(selected.id));
                    }
                }

                cv::Mat clippedFine;
                cv::min(cv::max(fine, -0.07f), 0.07f, clippedFine);
                const cv::Mat preClamp =
                    1.0f * mapped + 0.5f * clippedFine + 1.0f * structural;

                double minimum = 0.0, maximum = 0.0;
                cv::minMaxLoc(preClamp, &minimum, &maximum);
                const double pixels = static_cast<double>(base.total());

                excursions << selected.id << ',' << grid << ','
                           << minimum << ',' << maximum << ','
                           << cv::countNonZero(preClamp < 0.0f) / pixels << ','
                           << cv::countNonZero(preClamp > 1.0f) / pixels
                           << '\n';

                const std::string label = "legacy_grid" + std::to_string(grid);
                metrics(csv, selected.id, label.c_str(), result);
                save(destination / (label + ".png"), result);
                save(destination / (label + "__base.png"), mapped8);
            }

            std::cout << selected.id
                      << ": exact 64x64 baseline; 8x8 candidate exported\n";
        }

        csv.close();
        excursions.close();
        if (!csv || !excursions) {
            throw std::runtime_error("Cannot finalize reports");
        }

        std::ofstream complete(output / "COMPLETE.txt");
        complete << "cases=6\nmetric_rows=24\n"
                 << "baseline_exact_matches=6\n"
                 << "changed_parameter=CLAHE_tile_grid_64_to_8\n"
                 << "clip_limit=2\n"
                 << "additional_postprocessing=none\n"
                 << "opencv=" << CV_VERSION << '\n';
        complete.close();

        if (!complete) {
            throw std::runtime_error("Cannot finalize completion marker");
        }

        std::cout << "Complete: 6 cases; 24 rows\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what()
                  << "\nTreat this experiment as incomplete.\n";
        return 1;
    }
}