//
// Created by hakgu on 3/25/2026.
//
#include "rgf.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ref_rgf {

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

    cv::Mat var_I  = mean_II - mean_I.mul(mean_I);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a = box_filter(a, r) / area;
    cv::Mat mean_b = box_filter(b, r) / area;

    cv::Mat q = mean_a.mul(I) + mean_b;
    return clamp01(q);
}

cv::Mat rolling_guided_filter(const cv::Mat& gray8, int iters, int r, float eps, float sigma0) {
    CV_Assert(gray8.type() == CV_8U);

    cv::Mat p;
    gray8.convertTo(p, CV_32F, 1.0 / 255.0);

    cv::Mat guide;
    cv::GaussianBlur(p, guide, cv::Size(0, 0), sigma0, sigma0, cv::BORDER_REFLECT101);

    cv::Mat base = guide.clone();
    const int n = std::max(1, iters);

    for (int i = 0; i < n; ++i) {
        base = guided_filter_gray(guide, p, r, eps);
        guide = base;
    }

    return base;
}

void rgf_decompose(const cv::Mat& gray8, cv::Mat& base, cv::Mat& details,
                   int iters, int r, float eps, float sigma0) {
    CV_Assert(gray8.type() == CV_8U);

    base = rolling_guided_filter(gray8, iters, r, eps, sigma0);

    cv::Mat p;
    gray8.convertTo(p, CV_32F, 1.0 / 255.0);
    details = p - base;
}

void rgf_thermal(const cv::Mat& input_thermal, cv::Mat& thermal8, cv::Mat& base, cv::Mat& detail,
                 int iters, int r, float eps, float sigma0, float p_lo, float p_hi) {
    if (input_thermal.empty()) {
        throw std::runtime_error("Missing input");
    }

    if (input_thermal.type() != CV_8U) {
        thermal8 = percentile_to_8bit(input_thermal, p_lo, p_hi);
    } else {
        thermal8 = input_thermal.clone();
    }

    rgf_decompose(thermal8, base, detail, iters, r, eps, sigma0);
}

cv::Mat fuse_on_L(const cv::Mat& vis_bgr, const cv::Mat& guide_gray01, float a, float b, float c) {
    CV_Assert(!vis_bgr.empty());
    CV_Assert(vis_bgr.type() == CV_8UC3);
    CV_Assert(guide_gray01.type() == CV_32F);

    const int h = vis_bgr.rows;
    const int w = vis_bgr.cols;

    cv::Mat guide8;
    cv::Mat clipped = clamp01(guide_gray01);
    clipped.convertTo(guide8, CV_8U, 255.0);

    cv::Mat guide_resized;
    cv::resize(guide8, guide_resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

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

} // namespace ref_rgf