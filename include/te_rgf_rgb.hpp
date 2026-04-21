//
// Created by hakgu on 1/30/2026.
//

#ifndef RGB_THERMAL_FUSION_C_TE_RGF_RGB_HPP
#define RGB_THERMAL_FUSION_C_TE_RGF_RGB_HPP

#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>

namespace te {
    // Structs of parameters
    struct Scales_RGF {
        int iters = 4, r = 4;
        float eps = 1e-3f, sigma0 = 1.5f;
    };
    struct fusionParameters {
        int ksize_H = 7;
        float alpha_lumin = 0.6f, b_fine = 1.0f, b_struct = 1.0f, b_base = 1.0f;

        float percentile_lo = 1.0f;
        float percentile_hi = 99.0f;
    };
    struct fusionResult {
        cv::Mat fused_image;
        cv::Mat lumin_effect;
        std::map<std::string, cv::Mat> dictionary;
    };

    // Utilities
    cv::Mat percentile8bit(const cv::Mat& img, float p_lo=1.f, float p_high=99.f);
    cv::Mat box_filter(const cv::Mat& img, int radian);
    cv::Mat normalize01(const cv::Mat& img);

    cv::Mat local_entropy(const cv::Mat& gray8, int ksize=7, int bins=16);
    cv::Mat local_contrast(const cv::Mat& gray8, int ksize=7);

    cv::Mat clahe(const cv::Mat& base, double clip_limit=2.0, cv::Size tile_grid_size=cv::Size(64,64));
    cv::Mat build_luminance(const cv::Mat& detail_fine, const cv::Mat& detail_struct, const cv::Mat& enhanced_base,
        float beta_fine=1.0f, float beta_struct=1.0f, float beta_base=1.0f, float fine_clip=0.1f);
    cv::Mat luminance_fusion(const cv::Mat& vis_img, const cv::Mat& lumin, float a=0.6f);

    cv::Mat guided_filter_gray(const cv::Mat& guided_img, const cv::Mat& src_img, int r=6, float eps=1e-3f);
    cv::Mat rolling_guided_filter(const cv::Mat& thermal, int iters=4, int r=6, float eps=1e-3f, float sigma0=1.5f);
    std::vector<cv::Mat> rgf_multi_scale(const cv::Mat& img8, const std::vector<Scales_RGF>& v_scales);

    // Functions
    void weight_map(const cv::Mat& thermal, int ksize, float w_fine_min, float w_fine_max, float w_struct_min,
        float w_struct_max, float c_clip, cv::Mat& w_fine_out, cv::Mat& w_struct_out);
    void rgf_decompose(const cv::Mat& t8, const std::vector<Scales_RGF>& v_scales, cv::Mat& d1, cv::Mat& d2, cv::Mat& d3, cv::Mat& base_img);
    void enhance_details(const cv::Mat& d1, const cv::Mat& d2, const cv::Mat& d3, const cv::Mat& w_fine, const cv::Mat& w_struct,
        cv::Mat& d_fine_enhanced, cv::Mat& d_struct_enhanced, float fine_post=0.1f);

    fusionResult rgb_fusion(const cv::Mat& vis_img, const cv::Mat& thermal, const std::vector<Scales_RGF>* v_scales = nullptr, const fusionParameters& pmt = {});;
}

#endif //RGB_THERMAL_FUSION_C_TE_RGF_RGB_HPP