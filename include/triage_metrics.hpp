//
// Created by hakgu on 3/25/2026.
//

#ifndef RGB_THERMAL_FUSION__C___TRIAGE_METRICS_HPP
#define RGB_THERMAL_FUSION__C___TRIAGE_METRICS_HPP

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace triage {

    struct MetricSet {
        double entropy = 0.0;
        double sobel = 0.0;
        double lap_var = 0.0;
        double rms = 0.0;
    };

    cv::Mat read_gray(const std::string& path);
    cv::Mat to_gray_u8(const cv::Mat& img);

    double entropy(const cv::Mat& img8);
    double lap_var(const cv::Mat& img8);
    double sobel(const cv::Mat& img8);
    double rms(const cv::Mat& img8);

    MetricSet metrics(const cv::Mat& img);

    void ensure_dir(const std::string& path);
    void write_panel(const std::string& out_path,
                     const std::vector<std::pair<std::string, cv::Mat>>& images);

    std::vector<std::pair<std::string, std::string>> thermal_frames(const std::string& root);

} // namespace triage

#endif //RGB_THERMAL_FUSION__C___TRIAGE_METRICS_HPP