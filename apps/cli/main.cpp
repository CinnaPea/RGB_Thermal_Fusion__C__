#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<chrono>
#include<numeric>
#include<algorithm>
#include<thread>
#include "te_rgf_rgb.hpp"

static cv::Mat to_u8_01(const cv::Mat& input) {
    cv::Mat u8, temp = input.clone();
    cv::min(cv::max(temp, 0), 1, temp);
    temp *= 255.f;
    temp.convertTo(u8, CV_8U);
    return u8;
}

static double ms_since(const std::chrono::high_resolution_clock::time_point& t0,
                       const std::chrono::high_resolution_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    std::string outDirectory = (argc > 1) ? argv[1] : "RGB_Thermal_Fusion_C++";
    std::string visual = (argc > 2) ? argv[2] : "data/VIS3.jpg", thermal = (argc > 3) ? argv[3] : "data/IR3.jpg";

    // cv::utils::fs::createDirectories(outDir);
    cv::Mat vis = cv::imread(visual, cv::IMREAD_COLOR);
    cv::Mat ther = cv::imread(thermal, cv::IMREAD_ANYDEPTH);

    if (vis.empty() || ther.empty()) {
        std::cerr << "Missing images" << std::endl;
        return -1;
    }
    te::fusionParameters scales;
    scales.ksize_H = 7;
    scales.alpha_lumin = 0.75f;
    scales.b_fine = 0.55f;
    scales.b_struct = 0.8f;
    scales.b_base = 1.0f;

    // te::fusionResult r = te::rgb_fusion(vis, ther, nullptr, scales);

    cv::setUseOptimized(true);
    cv::setNumThreads(static_cast<int>(std::thread::hardware_concurrency()));

    std::cout << "VIS " << vis.cols << "x" << vis.rows
              << " IR " << ther.cols << "x" << ther.rows
              << " IR type=" << ther.type() << "\n";

    for (int i = 0; i < 3; ++i) {
        te::fusionResult warm = te::rgb_fusion(vis, ther, nullptr, scales);
        (void)warm;
    }

    constexpr int N = 100;
    std::vector<double> times;
    times.reserve(N);

    te::fusionResult r;
    for (int i = 0; i < N; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        r = te::rgb_fusion(vis, ther, nullptr, scales);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(ms_since(t0, t1));
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double p50 = sorted[sorted.size() / 2];
    double p95 = sorted[static_cast<size_t>(0.95 * (sorted.size() - 1))];
    double best = sorted.front();
    double worst = sorted.back();

    std::cout << "rgb_fusion timing over " << N << " runs:\n";
    std::cout << "  avg:  " << avg  << " ms (" << (1000.0 / avg) << " FPS)\n";
    std::cout << "  p50:  " << p50  << " ms\n";
    std::cout << "  p95:  " << p95  << " ms\n";
    std::cout << "  best: " << best << " ms\n";
    std::cout << "  worst: " << worst<< " ms\n";

    cv::imwrite("F:/RGB_Thermal_Fusion__C++/output/fused_rgb_thermal_C.png", r.fused_image);
    cv::imwrite("F:/RGB_Thermal_Fusion__C++/output/thermal_C.png", r.dictionary["thermal8"]);
    cv::imwrite("F:/RGB_Thermal_Fusion__C++/output/enhanced_base_rgb_C.png", to_u8_01(r.dictionary["B_enh"]));
    cv::imwrite("F:/RGB_Thermal_Fusion__C++/output/luminance_rgb_C.png", to_u8_01(r.dictionary["L_T"]));

    std::cout << "Success\n";
    return 0;
}