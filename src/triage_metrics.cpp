//
// Created by hakgu on 3/25/2026.
//
#include "triage_metrics.hpp"

#include <filesystem>
#include <regex>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

namespace triage {

cv::Mat read_gray(const std::string& path) {
    cv::Mat im = cv::imread(path, cv::IMREAD_ANYDEPTH);
    if (im.empty()) {
        throw std::runtime_error("File not found: " + path);
    }
    if (im.channels() == 3) {
        cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
    }
    return im;
}

cv::Mat to_gray_u8(const cv::Mat& img) {
    cv::Mat g;
    if (img.channels() == 3) {
        cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    } else {
        g = img.clone();
    }

    if (g.type() != CV_8U) {
        cv::normalize(g, g, 0, 255, cv::NORM_MINMAX);
        g.convertTo(g, CV_8U);
    }
    return g;
}

double entropy(const cv::Mat& img8) {
    CV_Assert(img8.type() == CV_8U);

    int histSize[] = {256};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    int channels[] = {0};

    cv::Mat hist;
    cv::calcHist(&img8, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    const double sum_hist = cv::sum(hist)[0] + 1e-12;
    double H = 0.0;

    for (int i = 0; i < 256; ++i) {
        const double p = hist.at<float>(i) / sum_hist;
        if (p > 0.0) {
            H -= p * std::log2(p);
        }
    }
    return H;
}

double lap_var(const cv::Mat& img8) {
    cv::Mat lap;
    cv::Laplacian(img8, lap, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

double sobel(const cv::Mat& img8) {
    cv::Mat gx, gy;
    cv::Sobel(img8, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img8, gy, CV_32F, 0, 1, 3);

    cv::Mat mag2 = gx.mul(gx) + gy.mul(gy);
    return cv::mean(mag2)[0];
}

double rms(const cv::Mat& img8) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(img8, mean, stddev);
    return stddev[0];
}

MetricSet metrics(const cv::Mat& img) {
    cv::Mat gray8 = to_gray_u8(img);
    MetricSet m;
    m.entropy = entropy(gray8);
    m.sobel = sobel(gray8);
    m.lap_var = lap_var(gray8);
    m.rms = rms(gray8);
    return m;
}

void ensure_dir(const std::string& path) {
    fs::create_directories(path);
}

void write_panel(const std::string& out_path,
                 const std::vector<std::pair<std::string, cv::Mat>>& images) {
    std::vector<cv::Mat> columns;
    int hmax = 0;

    for (const auto& [name, img] : images) {
        if (img.empty()) continue;

        cv::Mat disp;
        if (img.type() != CV_8U && img.type() != CV_8UC3) {
            cv::normalize(img, disp, 0, 255, cv::NORM_MINMAX);
            disp.convertTo(disp, CV_8U);
        } else {
            disp = img.clone();
        }

        cv::Mat bar;
        if (disp.channels() == 1) {
            bar = cv::Mat(disp.cols, 24, CV_8U);
            bar = cv::Mat(24, disp.cols, CV_8U, cv::Scalar(240));
            cv::putText(bar, name, cv::Point(6, 16), cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(0), 1, cv::LINE_AA);
        } else {
            bar = cv::Mat(24, disp.cols, CV_8UC3, cv::Scalar(240, 240, 240));
            cv::putText(bar, name, cv::Point(6, 16), cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        cv::Mat tile;
        cv::vconcat(bar, disp, tile);
        columns.push_back(tile);
        hmax = std::max(hmax, tile.rows);
    }

    if (columns.empty()) return;

    std::vector<cv::Mat> padded;
    for (auto& c : columns) {
        if (c.rows < hmax) {
            cv::Mat pad;
            if (c.channels() == 1) {
                pad = cv::Mat(hmax - c.rows, c.cols, CV_8U, cv::Scalar(255));
            } else {
                pad = cv::Mat(hmax - c.rows, c.cols, CV_8UC3, cv::Scalar(255, 255, 255));
            }
            cv::vconcat(c, pad, c);
        }
        padded.push_back(c);
    }

    cv::Mat out = padded[0];
    for (size_t i = 1; i < padded.size(); ++i) {
        cv::Mat gap;
        if (out.channels() == 1) {
            gap = cv::Mat(hmax, 10, CV_8U, cv::Scalar(255));
        } else {
            gap = cv::Mat(hmax, 10, CV_8UC3, cv::Scalar(255, 255, 255));
        }
        cv::hconcat(std::vector<cv::Mat>{out, gap, padded[i]}, out);
    }

    ensure_dir(fs::path(out_path).parent_path().string());
    cv::imwrite(out_path, out);
}

std::vector<std::pair<std::string, std::string>> thermal_frames(const std::string& root) {
    std::vector<std::pair<std::string, std::string>> frames;

    if (!fs::exists(root) || !fs::is_directory(root)) {
        return frames;
    }

    const std::regex pat(R"(^(\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$)", std::regex::icase);

    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;

        const std::string fname = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_match(fname, m, pat)) continue;

        std::string pid = m[1].str();
        pid.erase(0, std::min(pid.find_first_not_of('0'), pid.size() - 1));
        if (pid.empty()) pid = "0";

        frames.emplace_back(pid, entry.path().string());
    }

    std::sort(frames.begin(), frames.end(),
              [](const auto& a, const auto& b) {
                  return std::stoi(a.first) < std::stoi(b.first);
              });

    return frames;
}

} // namespace triage