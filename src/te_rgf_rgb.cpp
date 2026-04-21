//
// Created by hakgu on 1/30/2026.
//

#include "te_rgf_rgb.hpp"
#include<algorithm>
#include<cmath>
#include<stdexcept>
#include<numeric>
#include<opencv2/ximgproc/edge_filter.hpp>

namespace te {
    using namespace cv;

    static Mat clamp(const Mat& m) {
        Mat clamped;
        min(max(m, 0), 1, clamped);
        return clamped;
    }

    Mat percentile8bit(const Mat &img, const float p_lo, const float p_high) {
        if (img.empty()) throw std::runtime_error("Empty image");
        Mat processed_thermal;
        if (img.channels() == 3) {
            cvtColor(img, processed_thermal, COLOR_BGR2GRAY);
        } else if (img.channels() == 4) {
            cvtColor(img, processed_thermal, COLOR_BGRA2GRAY);
        } else {
            processed_thermal = img;
        }

        if (processed_thermal.type() == CV_8U) return processed_thermal.clone();

        Mat f;
        processed_thermal.convertTo(f, CV_32F);

        std::vector<float> v;
        v.reserve(f.total());
        if (f.isContinuous()) {
            const float* pointer = f.ptr<float>(0);
            v.assign(pointer, pointer + f.total());
        } else {
            for (int i = 0; i < f.rows; ++i) {
                const float* pointer_row = f.ptr<float>(i);
                v.insert(v.end(), pointer_row, pointer_row + f.cols);
            }
        }

        auto pick = [&](std::vector<float>& c, const float percentile) {
            const float position = percentile / 100.f * static_cast<float>(c.size() - 1);
            const size_t n = std::clamp(position, 0.f, static_cast<float>(c.size() - 1));
            std::nth_element(c.begin(), c.begin() + n, c.end());
            return c[n];
        };

        std::vector<float> v_2 = v;
        const float lo = pick(v, p_lo);
        float hi = pick(v_2, p_high);
        hi = std::max(hi, lo + 1e-6f);

        const Mat output = (f - lo) * (255.f / (hi - lo));
        Mat u8;
        output.convertTo(u8, CV_8U);
        return u8;
    }

    Mat box_filter(const Mat &img, const int radian) {
        const int k = 2 * radian + 1;
        Mat output;
        boxFilter(img, output, -1, Size(k, k), Point(-1, -1), false, BORDER_REFLECT101);
        return output;
    }

    Mat normalize01(const Mat &img) {
        double minVal, maxVal;
        minMaxLoc(img, &minVal, &maxVal);
        if (maxVal <= minVal + 1e-6) return Mat::zeros(img.size(), CV_32F);
        Mat xf;
        img.convertTo(xf, CV_32F);
        Mat output = (xf - static_cast<float>(minVal)) * (1.0f / static_cast<float>(maxVal - minVal));
        return output;
    }

    Mat local_entropy(const Mat &gray8, int ksize, int bins) {
        CV_Assert(ksize > 1 && ksize % 2 == 1);
        Mat gray;
        if (gray8.type() != CV_8U) {
            gray8.convertTo(gray, CV_8U);
        } else {
            gray = gray8;
        }

        Mat q(gray.size(), CV_8U);
        for (int i = 0; i < gray.rows; ++i) {
            const uint8_t* src = gray.ptr<uint8_t>(i);
            auto* dst = q.ptr<uint8_t>(i);
            for (int j = 0; j < gray.cols; ++j) {
                dst[j] = static_cast<uint8_t>((src[j] * bins) >> 8);
            }
        }

        const int area = ksize * ksize;

        thread_local Mat lut(1, 256, CV_32F);
        if (thread_local int lastArea = -1; area != lastArea || lut.empty()) {
            lut.setTo(0);
            auto* L = lut.ptr<float>(0);
            const float invArea = 1.0f / static_cast<float>(area);
            const float invLog2 = 1.0f / std::log(2.0f);

            const int maxC = std::min(area, 255);
            for (int c = 1; c <= maxC; ++c) {
                const float p = static_cast<float>(c) * invArea;
                L[c] = -(p * (std::log(p) * invLog2));
            }
            lastArea = area;
        }

        Mat H = Mat::zeros(gray.size(), CV_32F);

        thread_local Mat mask_u8, cnt16u, cnt8u, p;
        for (int b = 0; b < bins; ++b) {
            compare(q, b, mask_u8, CMP_EQ);
            boxFilter(mask_u8, cnt16u, CV_16U, Size(ksize, ksize),
                Point(-1, -1), false, BORDER_REFLECT101);
            cnt16u /= 255;
            cnt16u.convertTo(cnt8u, CV_8U);
            LUT(cnt8u, lut, p);
            H += p;
        }

        H *= (1.0f / std::log2(static_cast<float>(bins)));
        min(max(H, 0), 1, H);
        return H;
    }

    Mat local_contrast(const Mat &gray8, int ksize) {
        Mat gray, mean, mean2;
        if (gray8.type() != CV_8U) {
            gray8.convertTo(gray, CV_8U);
        } else {
            gray = gray8;
        }

        Mat g;
        gray.convertTo(g, CV_32F, 1.0 / 255.0);
        boxFilter(g, mean, -1, Size(ksize, ksize), Point(-1, -1), true, BORDER_REFLECT101);
        boxFilter(g.mul(g), mean2, -1, Size(ksize, ksize), Point(-1, -1), true, BORDER_REFLECT101);

        Mat var = mean2 - mean.mul(mean), stddev;
        max(var, 0, var);
        sqrt(var, stddev);

        stddev *= 2.0f;
        return normalize01(stddev);
    }

    void weight_map(const Mat &thermal, int ksize, float w_fine_min, float w_fine_max,
                    float w_struct_min, float w_struct_max, float c_clip,
                    Mat &w_fine_out, Mat &w_struct_out) {
        Mat H = local_entropy(thermal, ksize, 8);
        Mat C = local_contrast(thermal, ksize);
        Mat overload = (C - c_clip) * (1.f / (1.f - c_clip + 1e-6f));
        min(max(overload, 0), 1, overload);

        Mat w_fine   = 1.f + 0.8f * (H - 0.5f) - 0.5f * overload;
        Mat w_struct = 1.f + 0.6f * (H - 0.5f) + 0.6f * (C - 0.5f);

        // Existing overload suppression
        Mat gate_over = 1.f - 0.8f * overload;

        // New confidence gate: suppress fine enhancement in weak / noisy regions
        // Low C -> low confidence -> stronger suppression of fine detail
        Mat gate_conf = 0.35f + 0.65f * C;

        w_fine = w_fine.mul(gate_over);
        w_fine = w_fine.mul(gate_conf);

        min(max(w_fine, w_fine_min), w_fine_max, w_fine);
        min(max(w_struct, w_struct_min), w_struct_max, w_struct);

        w_fine_out = w_fine;
        w_struct_out = w_struct;
    }

    Mat guided_filter_gray(const Mat &guided_img, const Mat &src_img, const int r, const float eps) {
        Mat q;
        ximgproc::guidedFilter(guided_img, src_img, q, r, eps);
        min(max(q, 0), 1, q);
        return q;
    }

    Mat rolling_guided_filter(const Mat &thermal, const int iters, const int r, const float eps, const float sigma0) {
        CV_Assert(thermal.type() == CV_8U);
        Mat p, guide;
        thermal.convertTo(p, CV_32F, 1.0 / 255.0);
        GaussianBlur(p, guide, Size(0, 0), sigma0, sigma0, BORDER_REFLECT101);

        Mat base = guide.clone();
        const int n = std::max(1, iters);

        if (iters <= 1) {
            base = guided_filter_gray(guide, p, r, eps);
        } else {
            for (int i = 0; i < n; ++i) {
                base = guided_filter_gray(guide, p, r, eps);
                guide = base;
            }
        }
        return base;
    }

    std::vector<Mat> rgf_multi_scale(const Mat &img8, const std::vector<Scales_RGF> &v_scales) {
        std::vector<Mat> bases;
        bases.reserve(v_scales.size());
        for (const auto& [iters, r, eps, sigma0] : v_scales) {
            Mat b = rolling_guided_filter(img8, iters, r, eps, sigma0);
            bases.push_back(b);
        }
        return bases;
    }

    void rgf_decompose(const Mat &t8, const std::vector<Scales_RGF> &v_scales, Mat &d1, Mat &d2, Mat &d3, Mat &base_img) {
        if (static_cast<int>(v_scales.size()) != 3) throw std::runtime_error("Requires 3 scales");
        CV_Assert(t8.type() == CV_8U);

        Mat I;
        t8.convertTo(I, CV_32F, 1.0 / 255.0);
        const auto bases = rgf_multi_scale(t8, v_scales);

        const Mat P1 = bases[0], P2 = bases[1], P3 = bases[2];
        d1 = I - P1;
        min(max(d1, -0.15f), 0.15f, d1);
        d2 = P1 - P2;
        d3 = P2 - P3;
        base_img = P3;
    }

    void enhance_details(const Mat &d1, const Mat &d2, const Mat &d3,
                         const Mat &w_fine, const Mat &w_struct,
                         Mat &d_fine_enhanced, Mat &d_struct_enhanced,
                         const float fine_post) {
        const Mat D_structural = d2 + d3;
        d_fine_enhanced = d1.mul(w_fine);
        min(max(d_fine_enhanced, -fine_post), fine_post, d_fine_enhanced);
        d_struct_enhanced = D_structural.mul(w_struct);
    }

    Mat clahe(const Mat &base, const double clip_limit, const Size tile_grid_size) {
        CV_Assert(base.type() == CV_32F);
        Mat base8;
        const Mat temp = base * 255.f;
        temp.convertTo(base8, CV_8U);

        static thread_local Ptr<CLAHE> c;
        if (!c) c = createCLAHE(clip_limit, tile_grid_size);

        Mat equalizer, output;
        c->apply(base8, equalizer);
        equalizer.convertTo(output, CV_32F, 1.0 / 255.0);
        return output;
    }

    Mat build_luminance(const Mat &detail_fine, const Mat &detail_struct, const Mat &enhanced_base,
                        const float beta_fine, const float beta_struct,
                        const float beta_base, const float fine_clip) {
        Mat df;
        detail_fine.copyTo(df);
        min(max(df, -fine_clip), fine_clip, df);
        Mat lumin = beta_base * enhanced_base + beta_fine * df + beta_struct * detail_struct;
        lumin = clamp(lumin);
        return lumin;
    }

    Mat luminance_fusion(const Mat &vis_img, const Mat &lumin, float a) {
        if (vis_img.empty()) throw std::runtime_error("Missing input");
        CV_Assert(vis_img.type() == CV_8UC3 && lumin.type() == CV_32F);

        Mat lumin_resized;
        if (lumin.size() == vis_img.size())
            lumin_resized = lumin;
        else
            resize(lumin, lumin_resized, vis_img.size(), 0, 0, INTER_LINEAR);

        Mat lab;
        cvtColor(vis_img, lab, COLOR_BGR2YCrCb);

        Mat L8, L_visual, L_output, L_out8;
        extractChannel(lab, L8, 0);
        L8.convertTo(L_visual, CV_32F, 1.0 / 255.0);
        L_output = a * L_visual + (1.f - a) * lumin_resized;
        L_output = clamp(L_output);
        L_output.convertTo(L_out8, CV_8U, 255.0);
        insertChannel(L_out8, lab, 0);

        Mat fused_bgr;
        cvtColor(lab, fused_bgr, COLOR_YCrCb2BGR);
        return fused_bgr;
    }

    fusionResult rgb_fusion(const Mat &vis_img, const Mat &thermal,
                            const std::vector<Scales_RGF> *v_scales,
                            const fusionParameters &pmt) {
        TickMeter tm;
        auto log = [&](const char* name) {
            tm.stop();
            std::cout << name << ": " << tm.getTimeMilli() << " ms\n";
            tm.reset();
            tm.start();
        };
        tm.start();

        if (vis_img.empty() || thermal.empty()) throw std::runtime_error("Missing input");

        const float p_lo = std::clamp(pmt.percentile_lo, 0.0f, 99.9f);
        const float p_hi = std::clamp(pmt.percentile_hi, p_lo + 0.1f, 100.0f);

        Mat thermal8 = percentile8bit(thermal, p_lo, p_hi);
        log("Percentile");

        const std::vector<Scales_RGF> v_scale_structural{
            {1,4,1e-3f,1.2f},
            {1,4,1e-3f,2.0f},
            {1,4,1e-3f,3.0f}
        };
        const auto& scales = v_scales ? *v_scales : v_scale_structural;

        Mat D1, D2, D3, base_img;
        rgf_decompose(thermal8, scales, D1, D2, D3, base_img);
        log("decompose");

        Mat w_fine, w_struct;
        weight_map(thermal8, pmt.ksize_H, 0.3f, 1.0f, 0.8f, 1.4f, 0.5f, w_fine, w_struct);

        // Spatial stabilization for noisy thermal inputs
        GaussianBlur(w_fine,   w_fine,   Size(5,5), 1.0, 1.0, BORDER_REFLECT101);
        GaussianBlur(w_struct, w_struct, Size(5,5), 1.0, 1.0, BORDER_REFLECT101);
        log("weight_map");

        Mat D_fine_enhanced, D_struct_enhanced;
        enhance_details(D1, D2, D3, w_fine, w_struct, D_fine_enhanced, D_struct_enhanced, 0.07f);

        Mat base_enhanced = clahe(base_img, 2.0, Size(64,64));
        log("clahe");

        // Reduce fine-detail influence for better robustness on low-SNR thermal images
        Mat lumin_effect = build_luminance(
            D_fine_enhanced,
            D_struct_enhanced,
            base_enhanced,
            0.5f * pmt.b_fine,
            pmt.b_struct,
            pmt.b_base,
            0.07f
        );
        log("build_luminance");

        Mat fused_bgr = luminance_fusion(vis_img, lumin_effect, pmt.alpha_lumin);
        log("luminance_fusion");

        fusionResult output;
        output.fused_image = fused_bgr;
        output.lumin_effect = lumin_effect;

        output.dictionary["thermal8"] = thermal8;
        output.dictionary["D1"] = D1;
        output.dictionary["D2"] = D2;
        output.dictionary["D3"] = D3;
        output.dictionary["base"] = base_img;
        output.dictionary["w_fine"] = w_fine;
        output.dictionary["w_struct"] = w_struct;
        output.dictionary["D_fine_enh"] = D_fine_enhanced;
        output.dictionary["D_struct_enh"] = D_struct_enhanced;
        output.dictionary["B_enh"] = base_enhanced;
        output.dictionary["L_T"] = lumin_effect;

        return output;
    }

}