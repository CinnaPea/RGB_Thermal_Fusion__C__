//
// Created by hakgu on 3/25/2026.
//
#include "msgf.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ref_msgf {

namespace {

cv::Mat clamp01(const cv::Mat& m) {
    cv::Mat out;
    cv::min(cv::max(m, 0.0f), 1.0f, out);
    return out;
}

float percentile_from_float_mat(const cv::Mat& f32, float p) {
    std::vector<float> vals;
    vals.reserve(f32.total());

    if (f32.isContinuous()) {
        const float* ptr = f32.ptr<float>(0);
        vals.assign(ptr, ptr + f32.total());
    } else {
        for (int y = 0; y < f32.rows; ++y) {
            const float* row = f32.ptr<float>(y);
            vals.insert(vals.end(), row, row + f32.cols);
        }
    }

    if (vals.empty()) {
        throw std::runtime_error("Empty image data");
    }

    const float pos = (p / 100.0f) * static_cast<float>(vals.size() - 1);
    const size_t idx = static_cast<size_t>(std::clamp(pos, 0.0f, static_cast<float>(vals.size() - 1)));

    std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
    return vals[idx];
}

} // namespace

cv::Mat percentile_to_8bit(const cv::Mat& img, float p_lo, float p_hi) {
    if (img.empty()) {
        throw std::runtime_error("Missing input image");
    }

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = img;
    }

    if (gray.type() == CV_8U) {
        return gray.clone();
    }

    cv::Mat f32;
    gray.convertTo(f32, CV_32F);

    const float lo = percentile_from_float_mat(f32, p_lo);
    float hi = percentile_from_float_mat(f32, p_hi);
    hi = std::max(hi, lo + 1e-6f);

    cv::Mat scaled = (f32 - lo) * (255.0f / (hi - lo));
    cv::Mat clipped;
    cv::min(cv::max(scaled, 0.0f), 255.0f, clipped);

    cv::Mat out;
    clipped.convertTo(out, CV_8U);
    return out;
}

cv::Mat box_filter(const cv::Mat& img, int r) {
    const int k = 2 * r + 1;
    cv::Mat out;
    cv::boxFilter(img, out, -1, cv::Size(k, k), cv::Point(-1, -1), false, cv::BORDER_REFLECT101);
    return out;
}

cv::Mat guided_filter_gray(const cv::Mat& guide, const cv::Mat& src, int r, float eps) {
    cv::Mat I, p;
    guide.convertTo(I, CV_32F);
    src.convertTo(p, CV_32F);

    const float area = static_cast<float>((2 * r + 1) * (2 * r + 1));

    cv::Mat mean_I  = box_filter(I, r) / area;
    cv::Mat mean_p  = box_filter(p, r) / area;
    cv::Mat mean_II = box_filter(I.mul(I), r) / area;
    cv::Mat mean_Ip = box_filter(I.mul(p), r) / area;

    cv::Mat var_I = mean_II - mean_I.mul(mean_I);
    cv::max(var_I, 0.0f, var_I);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a = box_filter(a, r) / area;
    cv::Mat mean_b = box_filter(b, r) / area;

    cv::Mat q = mean_a.mul(I) + mean_b;
    return clamp01(q);
}

void MSGF_decompose_gray(const cv::Mat& thermal8,
                         cv::Mat& base,
                         std::vector<cv::Mat>& details,
                         const std::vector<int>& radii,
                         float eps,
                         float pre_sigma) {
    CV_Assert(thermal8.type() == CV_8U);

    cv::Mat t1;
    thermal8.convertTo(t1, CV_32F, 1.0 / 255.0);

    cv::Mat guide;
    cv::GaussianBlur(t1, guide, cv::Size(0, 0), pre_sigma, pre_sigma, cv::BORDER_DEFAULT);

    std::vector<cv::Mat> outs;
    outs.reserve(radii.size());

    for (int r : radii) {
        cv::Mat q = guided_filter_gray(guide, t1, r, eps);
        outs.push_back(q);
        guide = q;
    }

    if (outs.empty()) {
        throw std::runtime_error("MSGF_decompose_gray: radii is empty");
    }

    base = outs.back();
    details.clear();
    details.reserve(outs.size());

    cv::Mat d0 = t1 - outs[0];
    cv::min(cv::max(d0, -1.0f), 1.0f, d0);
    details.push_back(d0);

    for (size_t i = 1; i < outs.size(); ++i) {
        cv::Mat d = outs[i - 1] - outs[i];
        cv::min(cv::max(d, -1.0f), 1.0f, d);
        details.push_back(d);
    }
}

cv::Mat equalize_base_global_clahe(const cv::Mat& base01, double clip, int tiles) {
    CV_Assert(base01.type() == CV_32F);

    cv::Mat clipped = clamp01(base01);
    cv::Mat base8;
    clipped.convertTo(base8, CV_8U, 255.0);

    auto clahe = cv::createCLAHE(clip, cv::Size(tiles, tiles));
    cv::Mat beq8;
    clahe->apply(base8, beq8);

    cv::Mat out;
    beq8.convertTo(out, CV_32F, 1.0 / 255.0);
    return out;
}

cv::Mat fuse_on_L(const cv::Mat& vis_bgr, const cv::Mat& guide_gray01, float a, float b, float c) {
    CV_Assert(!vis_bgr.empty());
    CV_Assert(vis_bgr.type() == CV_8UC3);
    CV_Assert(guide_gray01.type() == CV_32F);

    cv::Mat guide8;
    clamp01(guide_gray01).convertTo(guide8, CV_8U, 255.0);

    cv::Mat guide_resized;
    cv::resize(guide8, guide_resized, vis_bgr.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat lab;
    cv::cvtColor(vis_bgr, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> ch;
    cv::split(lab, ch);

    cv::Mat fusedL;
    cv::addWeighted(ch[0], a, guide_resized, b, c, fusedL);

    ch[0] = fusedL;
    cv::merge(ch, lab);

    cv::Mat fused_bgr;
    cv::cvtColor(lab, fused_bgr, cv::COLOR_Lab2BGR);
    return fused_bgr;
}

void fuse_MSGF_visual(const cv::Mat& vis_bgr, const cv::Mat& input_thermal,
                      cv::Mat& fused,
                      cv::Mat& thermal8,
                      cv::Mat& base,
                      std::vector<cv::Mat>& details,
                      cv::Mat& base_equal,
                      cv::Mat& mix,
                      const std::vector<int>& radii,
                      float eps,
                      double base_clip,
                      int base_tiles,
                      const std::vector<float>& detail_gains,
                      float a,
                      float b) {
    thermal8 = percentile_to_8bit(input_thermal);
    MSGF_decompose_gray(thermal8, base, details, radii, eps);
    base_equal = equalize_base_global_clahe(base, base_clip, base_tiles);

    std::vector<float> gains = detail_gains;
    if (gains.empty()) {
        gains = {0.15f, 0.10f};
        while (gains.size() < details.size()) {
            gains.push_back(0.05f);
        }
    }

    mix = base_equal.clone();
    const size_t n = std::min(gains.size(), details.size());
    for (size_t i = 0; i < n; ++i) {
        mix += gains[i] * details[i];
    }
    mix = clamp01(mix);

    fused = fuse_on_L(vis_bgr, mix, a, b, 0.0f);
}

void msgf_thermal(const cv::Mat& input_thermal,
                  cv::Mat& thermal8,
                  cv::Mat& base,
                  std::vector<cv::Mat>& details,
                  cv::Mat& base_equal,
                  cv::Mat& mix,
                  const std::vector<int>& radii,
                  float eps,
                  double base_clip,
                  int base_tiles,
                  const std::vector<float>& detail_gains) {
    thermal8 = percentile_to_8bit(input_thermal);
    MSGF_decompose_gray(thermal8, base, details, radii, eps);
    base_equal = equalize_base_global_clahe(base, base_clip, base_tiles);

    std::vector<float> gains = detail_gains;
    if (gains.empty()) {
        gains = {0.15f, 0.10f};
        while (gains.size() < details.size()) {
            gains.push_back(0.05f);
        }
    }

    mix = base_equal.clone();
    const size_t n = std::min(gains.size(), details.size());
    for (size_t i = 0; i < n; ++i) {
        mix += gains[i] * details[i];
    }
    mix = clamp01(mix);
}

} // namespace ref_msgf