//
// Created by hakgu on 3/20/2026.
//

#ifndef RGB_THERMAL_FUSION_C_MAINWINDOW_HPP
#define RGB_THERMAL_FUSION__C___MAINWINDOW_HPP
#pragma once

#include <QWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QString>
#include <QTabWidget>
#include <QSlider>
#include <opencv2/opencv.hpp>

#include "te_rgf_rgb.hpp"
#include "triage_metrics.hpp"
#include "rgf.hpp"
#include "msgf.hpp"

class MainWindow final : public QWidget {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    void loadVisImg();
    void loadIrImg();
    void fuse();
    void runTriage();

    void clearVis();
    void clearIr();
    void clearResults();
    void clearTriage();
    void saveResults();
    void saveTriage();
    void updateTriage() const;

    static void updateLabel(QLabel* label, const cv::Mat& img);
    void updateStatus() const;
    void appendLog(const QString& msg) const;
    void appendTriageLog(const QString& msg) const;
    void updateTabs();

    static cv::Mat Displays(const cv::Mat& img);

    QPushButton* visBtn = nullptr;
    QPushButton* irBtn = nullptr;
    QPushButton* fusedBtn = nullptr;
    QPushButton* clearVisBtn = nullptr;
    QPushButton* clearIrBtn = nullptr;
    QPushButton* saveResultsBtn = nullptr;
    QPushButton* triageLoadBtn = nullptr;
    QPushButton* triageRunBtn = nullptr;
    QPushButton* triageSaveBtn = nullptr;
    QPushButton* triageViewBtn = nullptr;

    QLabel* visLabel = nullptr;
    QLabel* irLabel = nullptr;
    QLabel* statusLabel = nullptr;

    QPlainTextEdit* log = nullptr;
    QPlainTextEdit* triageLog = nullptr;
    QTabWidget* resultTab = nullptr;

    QLabel* fusedTab = nullptr;
    QLabel* irTab = nullptr;
    QLabel* baseTab = nullptr;
    QLabel* enhancedTab = nullptr;
    QLabel* detailsTab = nullptr;
    QLabel* compareInputTab = nullptr;
    QLabel* compareRGFTab = nullptr;
    QLabel* compareMSGFTab = nullptr;
    QLabel* compareHybridTab = nullptr;

    cv::Mat visImg;
    cv::Mat irImg;
    cv::Mat fusedImg;

    QString visPath;
    QString irPath;

    te::fusionResult final;
    bool hasResult = false;
    bool triageDiff = false;

    QLabel* percentileLoLabel = nullptr;
    QLabel* percentileHiLabel = nullptr;
    QSlider* percentileLoSlider = nullptr;
    QSlider* percentileHiSlider = nullptr;

    QTabWidget* mainTab = nullptr;
    QWidget* fusionPage = nullptr;
    QWidget* comparePage = nullptr;

    cv::Mat triageInputImg;
    cv::Mat triageRGF;
    cv::Mat triageMSGF;
    cv::Mat triageHybrid;
    cv::Mat triageDiffRGF;
    cv::Mat triageDiffMSGF;
};
#endif //RGB_THERMAL_FUSION_C_MAINWINDOW_HPP