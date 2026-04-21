//
// Created by hakgu on 3/25/2026.
//

#ifndef RGB_THERMAL_FUSION__C___MSGF_HPP
#define RGB_THERMAL_FUSION__C___MSGF_HPP

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace ref_msgf {

cv::Mat percentile_to_8bit(const cv::Mat& img, float p_lo = 1.0f, float p_hi = 99.0f);
cv::Mat box_filter(const cv::Mat& img, int r);
cv::Mat guided_filter_gray(const cv::Mat& guide, const cv::Mat& src, int r = 8, float eps = 1e-3f);

void MSGF_decompose_gray(const cv::Mat& thermal8,
                         cv::Mat& base,
                         std::vector<cv::Mat>& details,
                         const std::vector<int>& radii = {3, 8, 16, 32},
                         float eps = 1e-3f,
                         float pre_sigma = 1.2f);

cv::Mat equalize_base_global_clahe(const cv::Mat& base01, double clip = 2.0, int tiles = 8);

cv::Mat fuse_on_L(const cv::Mat& vis_bgr, const cv::Mat& guide_gray01,
                  float a = 0.6f, float b = 0.4f, float c = 0.0f);

void fuse_MSGF_visual(const cv::Mat& vis_bgr, const cv::Mat& input_thermal,
                      cv::Mat& fused,
                      cv::Mat& thermal8,
                      cv::Mat& base,
                      std::vector<cv::Mat>& details,
                      cv::Mat& base_equal,
                      cv::Mat& mix,
                      const std::vector<int>& radii = {3, 8, 16, 32},
                      float eps = 1e-3f,
                      double base_clip = 2.0,
                      int base_tiles = 8,
                      const std::vector<float>& detail_gains = {},
                      float a = 0.6f,
                      float b = 0.4f);

void msgf_thermal(const cv::Mat& input_thermal,
                  cv::Mat& thermal8,
                  cv::Mat& base,
                  std::vector<cv::Mat>& details,
                  cv::Mat& base_equal,
                  cv::Mat& mix,
                  const std::vector<int>& radii = {3, 8, 16, 32},
                  float eps = 1e-3f,
                  double base_clip = 2.0,
                  int base_tiles = 8,
                  const std::vector<float>& detail_gains = {});

} // namespace ref_msgf

#endif //RGB_THERMAL_FUSION__C___MSGF_HPP