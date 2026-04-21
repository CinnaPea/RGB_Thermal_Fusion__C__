//
// Created by hakgu on 3/25/2026.
//

#ifndef RGB_THERMAL_FUSION__C___RGF_HPP
#define RGB_THERMAL_FUSION__C___RGF_HPP

#pragma once

#include <opencv2/opencv.hpp>

namespace ref_rgf {

    cv::Mat percentile_to_8bit(const cv::Mat& img, float p_lo = 1.0f, float p_hi = 99.0f);
    cv::Mat box_filter(const cv::Mat& img, int r);
    cv::Mat guided_filter_gray(const cv::Mat& guide, const cv::Mat& src, int r = 6, float eps = 1e-3f);
    cv::Mat rolling_guided_filter(const cv::Mat& gray8, int iters = 4, int r = 6, float eps = 1e-3f, float sigma0 = 1.5f);

    void rgf_decompose(const cv::Mat& gray8, cv::Mat& base, cv::Mat& details,
                       int iters = 4, int r = 6, float eps = 1e-3f, float sigma0 = 1.5f);

    void rgf_thermal(const cv::Mat& input_thermal, cv::Mat& thermal8, cv::Mat& base, cv::Mat& detail,
                     int iters = 4, int r = 6, float eps = 1e-3f, float sigma0 = 1.5f,
                     float p_lo = 1.0f, float p_hi = 99.0f);

    cv::Mat fuse_on_L(const cv::Mat& vis_bgr, const cv::Mat& guide_gray01,
                      float a = 0.65f, float b = 0.35f, float c = 0.0f);

} // namespace ref_rgf
#endif //RGB_THERMAL_FUSION__C___RGF_HPP