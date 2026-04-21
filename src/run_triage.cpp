//
// "run_triage.cpp" Created by hakgu on 3/25/2026.
//
#include "rgf.hpp"
#include "msgf.hpp"
#include "triage_metrics.hpp"
#include "te_rgf_rgb.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;

struct Row {
    std::string image;

    double entropy_rgf, entropy_hybrid, entropy_msgf;
    double sobel_rgf, sobel_hybrid, sobel_msgf;
    double lap_rgf, lap_hybrid, lap_msgf;
    double rms_rgf, rms_hybrid, rms_msgf;

    double t_rgf, t_hybrid, t_msgf, t_total;
};

static constexpr bool WRITE_PANELS = true;
static constexpr int WARMUP = 10;

static double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static double std_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const double m = mean_of(v);
    double acc = 0.0;
    for (double x : v) {
        const double d = x - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(v.size()));
}

static void write_csv(const std::string& path, const std::vector<Row>& rows) {
    triage::ensure_dir(fs::path(path).parent_path().string());
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (!f) {
        throw std::runtime_error("Cannot open CSV for writing: " + path);
    }

    f << "Image,"
      << "Entropy_RGF,Entropy_Hybrid,Entropy_MSGF,"
      << "Laplacian_RGF,Laplacian_Hybrid,Laplacian_MSGF,"
      << "Sobel_RGF,Sobel_Hybrid,Sobel_MSGF,"
      << "RMS_RGF,RMS_Hybrid,RMS_MSGF,"
      << "TimeMs_RGF,TimeMs_Hybrid,TimeMs_MSGF,TimeMs_Total\n";

    f << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        f << r.image << ","
          << r.entropy_rgf << "," << r.entropy_hybrid << "," << r.entropy_msgf << ","
          << r.lap_rgf << "," << r.lap_hybrid << "," << r.lap_msgf << ","
          << r.sobel_rgf << "," << r.sobel_hybrid << "," << r.sobel_msgf << ","
          << r.rms_rgf << "," << r.rms_hybrid << "," << r.rms_msgf << ","
          << r.t_rgf << "," << r.t_hybrid << "," << r.t_msgf << "," << r.t_total << "\n";
    }
}

int main() {
    const std::string LLVIP_IR = "F:/RGB_Thermal_Fusion__C++/data/LLVIP/set_1000";
    const std::string OUTROOT = "F:/RGB_Thermal_Fusion__C++/output/triage";
    const std::string PANEL   = OUTROOT + "/panel";
    const std::string CSV     = OUTROOT + "/csv/metrics_LLVIP_trio.csv";

    triage::ensure_dir(PANEL);
    triage::ensure_dir(fs::path(CSV).parent_path().string());

    auto frames = triage::thermal_frames(LLVIP_IR);
    if (frames.empty()) {
        std::cerr << "[FATAL] No thermal frames detected in: " << LLVIP_IR << "\n";
        return 1;
    }

    std::cout << "[DATA] found " << frames.size() << " thermal frames\n";

    std::vector<Row> rows;
    rows.reserve(frames.size());

    std::vector<double> t_rgf_list, t_hybrid_list, t_msgf_list, t_total_list;

    for (size_t idx = 0; idx < frames.size(); ++idx) {
        const auto& [pid, path_ir] = frames[idx];
        std::cout << "id = " << pid << " " << fs::path(path_ir).filename().string() << "\n";

        cv::Mat ir = triage::read_gray(path_ir);

        // --- RGF reference
        cv::Mat rgf_thermal8, rgf_base, rgf_detail;
        auto t0 = std::chrono::high_resolution_clock::now();
        ref_rgf::rgf_thermal(ir, rgf_thermal8, rgf_base, rgf_detail, 4, 6, 1e-3f, 1.5f, 1.0f, 99.0f);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double t_rgf = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cv::Mat rgf_u8;
        cv::Mat rgf_base_clipped;
        cv::min(cv::max(rgf_base, 0.0f), 1.0f, rgf_base_clipped);
        rgf_base_clipped.convertTo(rgf_u8, CV_8U, 255.0);

        // --- Hybrid current module
        cv::Mat fake_rgb;
        cv::cvtColor(rgf_thermal8, fake_rgb, cv::COLOR_GRAY2BGR);

        t0 = std::chrono::high_resolution_clock::now();
        te::fusionParameters pmt;
        auto hybrid_result = te::rgb_fusion(fake_rgb, ir, nullptr, pmt);
        t1 = std::chrono::high_resolution_clock::now();
        const double t_hybrid = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cv::Mat hybrid_L = hybrid_result.dictionary["L_T"];
        cv::Mat hybrid_clipped, hybrid_u8;
        cv::min(cv::max(hybrid_L, 0.0f), 1.0f, hybrid_clipped);
        hybrid_clipped.convertTo(hybrid_u8, CV_8U, 255.0);

        // --- MSGF reference
        cv::Mat msgf_thermal8, msgf_base, msgf_base_equal, msgf_mix;
        std::vector<cv::Mat> msgf_details;

        t0 = std::chrono::high_resolution_clock::now();
        ref_msgf::msgf_thermal(ir, msgf_thermal8, msgf_base, msgf_details, msgf_base_equal, msgf_mix,
                               {3, 8, 16, 32}, 1e-3f, 2.0, 8, {0.18f, 0.12f, 0.06f, 0.03f});
        t1 = std::chrono::high_resolution_clock::now();
        const double t_msgf = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cv::Mat msgf_u8;
        cv::Mat msgf_clipped;
        cv::min(cv::max(msgf_mix, 0.0f), 1.0f, msgf_clipped);
        msgf_clipped.convertTo(msgf_u8, CV_8U, 255.0);

        const double t_total = t_rgf + t_hybrid + t_msgf;

        // --- Metrics
        const auto m_rgf = triage::metrics(rgf_u8);
        const auto m_hybrid = triage::metrics(hybrid_u8);
        const auto m_msgf = triage::metrics(msgf_u8);

        rows.push_back(Row{
            pid,
            m_rgf.entropy, m_hybrid.entropy, m_msgf.entropy,
            m_rgf.sobel, m_hybrid.sobel, m_msgf.sobel,
            m_rgf.lap_var, m_hybrid.lap_var, m_msgf.lap_var,
            m_rgf.rms, m_hybrid.rms, m_msgf.rms,
            t_rgf, t_hybrid, t_msgf, t_total
        });

        if (static_cast<int>(idx) >= WARMUP) {
            t_rgf_list.push_back(t_rgf);
            t_hybrid_list.push_back(t_hybrid);
            t_msgf_list.push_back(t_msgf);
            t_total_list.push_back(t_total);
        }

        if (WRITE_PANELS) {
            const std::string panel_path = PANEL + "/IR" + pid + ".png";
            triage::write_panel(panel_path, {
                {"Input_t8", rgf_thermal8},
                {"RGF_base", rgf_u8},
                {"Hybrid_L", hybrid_u8},
                {"MSGF_mix", msgf_u8}
            });
        }
    }

    write_csv(CSV, rows);

    std::cout << "[ThermalCompare] Rows written: " << rows.size() << "\n";
    std::cout << "[ThermalCompare] CSV  -> " << CSV << "\n";
    std::cout << "[ThermalCompare] Panels -> " << PANEL << "\n";

    std::cout << "[TIME] warm-up excluded: first " << WARMUP << " frames\n";
    std::cout << "[TIME] RGF: mean=" << mean_of(t_rgf_list) << " ms, std=" << std_of(t_rgf_list)
              << " ms, n=" << t_rgf_list.size() << "\n";
    std::cout << "[TIME] Hybrid: mean=" << mean_of(t_hybrid_list) << " ms, std=" << std_of(t_hybrid_list)
              << " ms, n=" << t_hybrid_list.size() << "\n";
    std::cout << "[TIME] MSGF: mean=" << mean_of(t_msgf_list) << " ms, std=" << std_of(t_msgf_list)
              << " ms, n=" << t_msgf_list.size() << "\n";
    std::cout << "[TIME] Total: mean=" << mean_of(t_total_list) << " ms, std=" << std_of(t_total_list)
              << " ms, n=" << t_total_list.size() << "\n";

    return 0;
}