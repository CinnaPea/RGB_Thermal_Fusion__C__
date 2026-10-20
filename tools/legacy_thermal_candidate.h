//
// Created by hakgu on 10/17/2026.
//

#ifndef RGB_THERMAL_FUSION__C___LEGACY_THERMAL_CANDIDATE_H
#define RGB_THERMAL_FUSION__C___LEGACY_THERMAL_CANDIDATE_H

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace qart::experiment {

    class LegacyThermalCandidate final
    {
    public:
        LegacyThermalCandidate();

        LegacyThermalCandidate(
            const LegacyThermalCandidate&) = delete;

        LegacyThermalCandidate& operator=(
            const LegacyThermalCandidate&) = delete;

        cv::Mat enhance(
    const cv::Mat& plane,
    const std::string& tracePath = {});

    private:
        cv::Ptr<cv::CLAHE> clahe_;
    };

} // namespace qart::experiment

#endif //RGB_THERMAL_FUSION__C___LEGACY_THERMAL_CANDIDATE_H