//
// Created by hakgu on 10/17/2026.
//

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "legacy_thermal_candidate.h"

namespace {

void require(
    bool condition,
    const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireEqual(
    const cv::Mat& actual,
    const cv::Mat& expected,
    const std::string& context)
{
    require(
        actual.size() == expected.size()
        && actual.type() == expected.type(),
        context + ": geometry/type mismatch"
    );

    require(
        cv::norm(
            actual,
            expected,
            cv::NORM_INF
        ) == 0.0,
        context + ": pixels differ"
    );
}

int benchmarkLegacy(
    const std::filesystem::path& fixtureDirectory,
    const std::filesystem::path& outputDirectory)
{
#ifndef NDEBUG
    throw std::runtime_error(
        "Run the runtime measurement in Release"
    );
#endif

    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;

    constexpr int warmupCount = 2;
    constexpr int repeatCount = 5;
    constexpr int threadCount = 8;

    require(
        fs::is_directory(fixtureDirectory),
        "Fixture directory does not exist"
    );

    require(
        !fs::exists(outputDirectory),
        "Output exists; choose a fresh directory"
    );

    const std::string suffix = "__input.png";
    std::vector<fs::path> paths;

    for (
        const auto& entry :
        fs::directory_iterator(fixtureDirectory)
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
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());

    require(
        paths.size() == 40,
        "Expected exactly 40 native input PNGs"
    );

    struct Input
    {
        std::string id;
        cv::Mat image;
    };

    std::vector<Input> inputs;
    inputs.reserve(paths.size());

    // Decode all inputs before timing any processing.
    for (const auto& path : paths) {
        const std::string filename =
            path.filename().string();

        const std::string id = filename.substr(
            0,
            filename.size() - suffix.size()
        );

        require(
            id.find_first_of(",\t\r\n")
                == std::string::npos,
            "Invalid report case ID"
        );

        cv::Mat image = cv::imread(
            path.string(),
            cv::IMREAD_UNCHANGED
        );

        require(
            !image.empty()
            && image.dims == 2
            && image.type() == CV_8UC1,
            "Expected grayscale uint8 PNG: "
                + path.string()
        );

        inputs.push_back({
            id,
            std::move(image)
        });
    }

    fs::create_directories(outputDirectory);

    std::ofstream samples(
        outputDirectory / "runtime.csv"
    );

    std::ofstream metadata(
        outputDirectory / "environment.txt"
    );

    std::ofstream buildInformation(
        outputDirectory / "opencv_build.txt"
    );

    require(
        samples.good()
        && metadata.good()
        && buildInformation.good(),
        "Cannot open runtime reports"
    );

    cv::setNumThreads(threadCount);

    metadata
        << "method=legacy_grid8_fine1\n"
        << "build=Release\n"
        << "OpenCV=" << CV_VERSION << '\n'
        << "OpenCV_threads="
        << cv::getNumThreads() << '\n'
        << "OpenCV_optimized="
        << cv::useOptimized() << '\n'
        << "cases=" << inputs.size() << '\n'
        << "warmups_per_case=" << warmupCount << '\n'
        << "measured_calls_per_case="
        << repeatCount << '\n'
        << "timed_scope=enhance_only\n"
        << "excluded=decode,verification,"
        << "report_IO,rendering,constructor\n"
        << "order=sorted_case_major\n"
        << "tracing=disabled\n";

#ifdef __VERSION__
    metadata
        << "compiler=" << __VERSION__ << '\n';
#elif defined(_MSC_VER)
    metadata
        << "MSVC=" << _MSC_VER << '\n';
#endif

    buildInformation
        << cv::getBuildInformation();

    samples << std::setprecision(17);

    samples
        << "case_id,rows,columns,pixels,"
        << "repeat,milliseconds\n";

    qart::experiment::LegacyThermalCandidate
        enhancer;

    std::size_t measuredCalls = 0;
    double checksum = 0.0;

    for (const auto& input : inputs) {
        const cv::Mat original =
            input.image.clone();

        cv::Mat expected;

        for (
            int warmup = 0;
            warmup < warmupCount;
            ++warmup
        ) {
            cv::Mat result =
                enhancer.enhance(input.image);

            if (warmup == 0) {
                expected = result.clone();
            }
            else {
                requireEqual(
                    result,
                    expected,
                    input.id + ": warm-up repeatability"
                );
            }
        }

        for (
            int repeat = 0;
            repeat < repeatCount;
            ++repeat
        ) {
            cv::Mat result;

            const auto start = Clock::now();

            result = enhancer.enhance(
                input.image
            );

            const auto finish = Clock::now();

            const double milliseconds =
                std::chrono::duration<
                    double,
                    std::milli
                >(finish - start).count();

            require(
                std::isfinite(milliseconds)
                && milliseconds >= 0.0,
                "Invalid timing measurement"
            );

            // Verification is outside the measured interval.
            requireEqual(
                result,
                expected,
                input.id + ": measured output"
            );

            requireEqual(
                input.image,
                original,
                input.id + ": source preservation"
            );

            checksum += cv::sum(result)[0];

            samples
                << input.id << ','
                << input.image.rows << ','
                << input.image.cols << ','
                << input.image.total() << ','
                << repeat << ','
                << milliseconds << '\n';

            ++measuredCalls;
        }

        std::cout
            << input.id
            << ": 5 measured calls verified\n";
    }

    require(
        measuredCalls == 200,
        "Expected exactly 200 runtime rows"
    );

    metadata
        << std::setprecision(17)
        << "checksum=" << checksum << '\n';

    samples.close();
    metadata.close();
    buildInformation.close();

    require(
        !samples.fail()
        && !metadata.fail()
        && !buildInformation.fail(),
        "Runtime report write failed"
    );

    std::ofstream complete(
        outputDirectory / "COMPLETE.txt"
    );

    complete
        << "cases=40\n"
        << "measured_calls=200\n"
        << "warmups_per_case=2\n"
        << "output_repeatability=passed\n"
        << "source_preservation=passed\n"
        << "runtime_acceptance=pending_review\n";

    complete.close();

    require(
        !complete.fail(),
        "Cannot write completion marker"
    );

    std::cout
        << "Runtime complete: 40 cases; "
        << "200 measured calls\n";

    return 0;
}

void requireInvalid(
    const std::function<void()>& operation,
    const std::string& context)
{
    bool rejected = false;

    try {
        operation();
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }

    require(
        rejected,
        context + ": expected std::invalid_argument"
    );
}

cv::Mat makeRamp(int rows, int columns)
{
    cv::Mat image(rows, columns, CV_8UC1);

    for (int row = 0; row < rows; ++row) {
        auto* pixels = image.ptr<unsigned char>(row);

        for (
            int column = 0;
            column < columns;
            ++column
        ) {
            pixels[column] =
                static_cast<unsigned char>(
                    (row * 17 + column * 29) % 256
                );
        }
    }

    return image;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        cv::setNumThreads(8);

        qart::experiment::LegacyThermalCandidate
            enhancer;

        if (argc != 1) {
            if (
                argc == 4
                && std::string(argv[1]) == "--benchmark"
            ) {
                return benchmarkLegacy(
                    std::filesystem::path(argv[2]),
                    std::filesystem::path(argv[3])
                );
            }

            if (
                argc == 4
                && std::string(argv[1]) == "--trace"
            ) {
                const cv::Mat input = cv::imread(
                    argv[2],
                    cv::IMREAD_UNCHANGED
                );

                const cv::Mat output = enhancer.enhance(
                    input,
                    argv[3]
                );

                require(
                    !output.empty(),
                    "Trace execution returned empty output"
                );

                std::cout << "Trace written\n";
                return 0;
            }

            throw std::invalid_argument(
                "Usage: test_legacy_thermal_contract\n"
                "  [no arguments: contract tests]\n"
                "  --trace INPUT.png OUTPUT.yml.gz\n"
                "  --benchmark FIXTURE_DIR OUTPUT_DIR"
            );
        }

        requireInvalid(
            [&] {
                enhancer.enhance(cv::Mat{});
            },
            "Empty input"
        );

        requireInvalid(
            [&] {
                enhancer.enhance(
                    cv::Mat(
                        16, 16,
                        CV_8UC3,
                        cv::Scalar::all(0)
                    )
                );
            },
            "Three-channel input"
        );

        requireInvalid(
            [&] {
                enhancer.enhance(
                    cv::Mat(
                        16, 16,
                        CV_16UC1,
                        cv::Scalar(0)
                    )
                );
            },
            "uint16 input"
        );

        requireInvalid(
            [&] {
                enhancer.enhance(
                    cv::Mat(
                        16, 16,
                        CV_32FC1,
                        cv::Scalar(0)
                    )
                );
            },
            "Float input"
        );

        requireInvalid(
            [&] {
                const int dimensions[] = {
                    2, 3, 4
                };

                enhancer.enhance(
                    cv::Mat(
                        3,
                        dimensions,
                        CV_8UC1,
                        cv::Scalar(0)
                    )
                );
            },
            "Three-dimensional input"
        );

        std::vector<
            std::pair<std::string, cv::Mat>
        > cases;

        for (int value : {0, 1, 127, 255}) {
            cases.emplace_back(
                "constant_" + std::to_string(value),
                cv::Mat(
                    32, 32,
                    CV_8UC1,
                    cv::Scalar(value)
                )
            );
        }

        cases.emplace_back(
            "single_pixel",
            cv::Mat(
                1, 1,
                CV_8UC1,
                cv::Scalar(127)
            )
        );

        cases.emplace_back(
            "row",
            makeRamp(1, 33)
        );

        cases.emplace_back(
            "column",
            makeRamp(33, 1)
        );

        cases.emplace_back(
            "odd_size",
            makeRamp(31, 47)
        );

        cases.emplace_back(
            "ramp",
            makeRamp(64, 64)
        );

        cv::Mat hardEdge(
            64, 64,
            CV_8UC1,
            cv::Scalar(32)
        );

        hardEdge.colRange(32, 64).setTo(224);

        cases.emplace_back(
            "hard_edge",
            hardEdge
        );

        cv::Mat impulse(
            65, 65,
            CV_8UC1,
            cv::Scalar(0)
        );

        impulse.at<unsigned char>(32, 32) = 255;

        cases.emplace_back(
            "impulse",
            impulse
        );

        cv::Mat noise(63, 67, CV_8UC1);
        cv::RNG random(1701);

        random.fill(
            noise,
            cv::RNG::UNIFORM,
            0,
            256
        );

        cases.emplace_back(
            "noise",
            noise
        );

        const cv::Mat interveningInput =
            makeRamp(19, 23);

        for (const auto& item : cases) {
            const auto& name = item.first;
            const auto& input = item.second;

            const cv::Mat original =
                input.clone();

            cv::Mat first =
                enhancer.enhance(input);

            require(
                first.dims == 2
                && first.type() == CV_8UC1
                && first.size() == input.size(),
                name + ": invalid output contract"
            );

            requireEqual(
                input,
                original,
                name + ": source preservation"
            );

            const cv::Mat expected =
                first.clone();

            // Reusing the same instance must not alter
            // an already returned output.
            const cv::Mat unrelated =
                enhancer.enhance(interveningInput);

            require(
                !unrelated.empty(),
                name + ": intervening call failed"
            );

            requireEqual(
                first,
                expected,
                name + ": returned output lifetime"
            );

            requireEqual(
                enhancer.enhance(input),
                expected,
                name + ": repeated-call determinism"
            );

            // Overwrite every output pixel to detect
            // accidental aliasing with the source.
            first.setTo(255);

            requireEqual(
                input,
                original,
                name + ": output ownership"
            );

            requireEqual(
                enhancer.enhance(input),
                expected,
                name + ": caller output mutation"
            );

            std::cout
                << name << ": PASS\n";
        }

        // A source ROI must behave like its independent
        // contiguous copy.
        cv::Mat parent = makeRamp(51, 71);

        const cv::Mat parentBefore =
            parent.clone();

        const cv::Mat roi = parent(
            cv::Rect(5, 7, 43, 29)
        );

        require(
            !roi.isContinuous(),
            "ROI test must be non-contiguous"
        );

        requireEqual(
            enhancer.enhance(roi),
            enhancer.enhance(roi.clone()),
            "Non-contiguous input"
        );

        requireEqual(
            parent,
            parentBefore,
            "ROI parent preservation"
        );

        std::cout
            << "PASS: invalid-input rejection; "
            << cases.size()
            << " image cases; ROI equivalence; "
            << "ownership and repeatability\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Contract failed: "
            << error.what()
            << '\n';

        return 1;
    }
}