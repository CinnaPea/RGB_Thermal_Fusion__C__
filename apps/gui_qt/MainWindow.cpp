//
// Created by hakgu on 3/20/2026.
//

#include "MainWindow.hpp"

#include <QHBoxLayout>
#include <QGridLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <QFileInfo>
#include <QRegularExpression>
#include <QFrame>
#include <QSizePolicy>
#include <QElapsedTimer>
#include <QDateTime>
#include <Qt>
#include <QMap>
#include <functional>
#include <QDir>

#include <algorithm>

namespace {
    const QString kDataDir = "F:/RGB_Thermal_Fusion__C++/data";

    QImage matToQImage(const cv::Mat& mat) {
        if (mat.empty()) {
            return {};
        }

        switch (mat.type()) {
            case CV_8UC1: {
                return QImage(
                    mat.data,
                    mat.cols,
                    mat.rows,
                    static_cast<int>(mat.step),
                    QImage::Format_Grayscale8
                ).copy();
            }

            case CV_8UC3: {
                cv::Mat rgb;
                cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
                return QImage(
                    rgb.data,
                    rgb.cols,
                    rgb.rows,
                    static_cast<int>(rgb.step),
                    QImage::Format_RGB888
                ).copy();
            }

            case CV_8UC4: {
                cv::Mat rgba;
                cv::cvtColor(mat, rgba, cv::COLOR_BGRA2RGBA);
                return QImage(
                    rgba.data,
                    rgba.cols,
                    rgba.rows,
                    static_cast<int>(rgba.step),
                    QImage::Format_RGBA8888
                ).copy();
            }

            default:
                return {};
        }
    }

    QLabel* makeImageLabel(const QString& text) {
        auto* label = new QLabel(text);
        label->setAlignment(Qt::AlignCenter);
        label->setMinimumSize(320, 360);
        label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        label->setFrameShape(QFrame::Box);
        label->setStyleSheet(
            "QLabel {"
            "background-color: #1e1e1e;"
            "color: #cfcfcf;"
            "border: 1px solid #555;"
            "font-size: 14px;"
            "}"
        );
        return label;
    }

    QLabel* makeInfoLabel(const QString& text) {
        auto* label = new QLabel(text);
        label->setStyleSheet(
            "QLabel {"
            "color: #cfcfcf;"
            "font-size: 13px;"
            "padding: 4px;"
            "}"
        );
        return label;
    }

    int extractIndex(const QString& filePath, const QString& prefix) {
        const QString base = QFileInfo(filePath).completeBaseName();
        const QRegularExpression pattern(
            "^" + QRegularExpression::escape(prefix) + "(\\d+)$",
            QRegularExpression::CaseInsensitiveOption
        );

        const QRegularExpressionMatch match = pattern.match(base);
        if (!match.hasMatch()) {
            return -1;
        }

        bool ok = false;
        const int index = match.captured(1).toInt(&ok);
        return ok ? index : -1;
    }

    QString fileOnly(const QString& path) {
        return QFileInfo(path).fileName();
    }

    QString findCounterpartByIndex(const QString& sourcePath, const QString& targetPrefix) {
        const QFileInfo srcInfo(sourcePath);
        const QString sourcePrefix = targetPrefix.compare("IR", Qt::CaseInsensitive) == 0 ? "VIS" : "IR";
        const int index = extractIndex(sourcePath, sourcePrefix);
        if (index < 0) {
            return {};
        }

        const QString targetBase = QString("%1%2").arg(targetPrefix).arg(index);
        const QDir dir = srcInfo.dir();

        const QStringList filters = {
            targetBase + ".png",
            targetBase + ".jpg",
            targetBase + ".jpeg",
            targetBase + ".bmp",
            targetBase + ".tif",
            targetBase + ".tiff"
        };

        const QFileInfoList matches = dir.entryInfoList(filters, QDir::Files | QDir::Readable, QDir::Name);
        if (!matches.isEmpty()) {
            return matches.first().absoluteFilePath();
        }

        return {};
    }

    bool loadImageChecked(const QString& fileName, const int imreadFlag, cv::Mat& outImg) {
        const cv::Mat img = cv::imread(fileName.toStdString(), imreadFlag);
        if (img.empty()) {
            return false;
        }
        outImg = img;
        return true;
    }

    QString metricsToQString(const QString& name, const triage::MetricSet& m) {
        return QString("%1 | Entropy=%2 | Sobel=%3 | LapVar=%4 | RMS=%5")
            .arg(name)
            .arg(m.entropy, 0, 'f', 4)
            .arg(m.sobel, 0, 'f', 4)
            .arg(m.lap_var, 0, 'f', 4)
            .arg(m.rms, 0, 'f', 4);
    }
}

MainWindow::MainWindow(QWidget* parent) : QWidget(parent) {
    setWindowTitle("QT Image Fusion");
    resize(1400, 860);

    // =========================
    // Fusion page widgets
    // =========================
    visBtn = new QPushButton("Load VIS");
    irBtn = new QPushButton("Load IR");
    fusedBtn = new QPushButton("Fuse");
    saveResultsBtn = new QPushButton("Save");

    clearVisBtn = new QPushButton("X");
    clearIrBtn = new QPushButton("X");

    visLabel = makeImageLabel("Visible image");
    irLabel = makeImageLabel("Thermal image");

    fusedTab = makeImageLabel("Fused");
    irTab = makeImageLabel("Thermal 8-bit");
    baseTab = makeImageLabel("Enhanced Base");
    enhancedTab = makeImageLabel("Enhanced Thermal");
    detailsTab = makeImageLabel("Enhanced Details");

    resultTab = new QTabWidget;
    resultTab->addTab(fusedTab, "Fused");
    resultTab->addTab(irTab, "Thermal");
    resultTab->addTab(baseTab, "Base");
    resultTab->addTab(enhancedTab, "Thermal+");
    resultTab->addTab(detailsTab, "Details");

    statusLabel = new QLabel("VIS: - | IR: - | Not ready");
    statusLabel->setAlignment(Qt::AlignCenter);
    statusLabel->setStyleSheet(
        "QLabel {"
        "color: #cfcfcf;"
        "font-size: 13px;"
        "padding: 6px;"
        "}"
    );

    percentileLoSlider = new QSlider(Qt::Horizontal);
    percentileLoSlider->setRange(0, 100);
    percentileLoSlider->setValue(1);
    percentileLoSlider->setSingleStep(1);
    percentileLoSlider->setPageStep(1);

    percentileHiSlider = new QSlider(Qt::Horizontal);
    percentileHiSlider->setRange(90, 100);
    percentileHiSlider->setValue(99);
    percentileHiSlider->setSingleStep(1);
    percentileHiSlider->setPageStep(1);

    percentileLoLabel = makeInfoLabel("Low: 1.0");
    percentileHiLabel = makeInfoLabel("High: 99.0");

    auto* percentileLoTitle = makeInfoLabel("Thermal Percentile Low");
    auto* percentileHiTitle = makeInfoLabel("Thermal Percentile High");

    auto* percentileLoRow = new QHBoxLayout;
    percentileLoRow->addWidget(percentileLoTitle);
    percentileLoRow->addWidget(percentileLoSlider, 1);
    percentileLoRow->addWidget(percentileLoLabel);

    auto* percentileHiRow = new QHBoxLayout;
    percentileHiRow->addWidget(percentileHiTitle);
    percentileHiRow->addWidget(percentileHiSlider, 1);
    percentileHiRow->addWidget(percentileHiLabel);

    log = new QPlainTextEdit;
    log->setReadOnly(true);
    log->setMaximumBlockCount(500);
    log->setMinimumHeight(140);
    log->setStyleSheet(
        "QPlainTextEdit {"
        "background-color: #111111;"
        "color: #d8d8d8;"
        "border: 1px solid #555;"
        "font-family: Consolas, monospace;"
        "font-size: 12px;"
        "}"
    );

    for (QPushButton* btn : {visBtn, irBtn, fusedBtn, saveResultsBtn}) {
        btn->setMinimumHeight(36);
    }

    for (QPushButton* btn : {clearVisBtn, clearIrBtn}) {
        btn->setFixedSize(32, 32);
    }

    auto* visHeader = new QHBoxLayout;
    visHeader->addWidget(visBtn);
    visHeader->addWidget(clearVisBtn);

    auto* irHeader = new QHBoxLayout;
    irHeader->addWidget(irBtn);
    irHeader->addWidget(clearIrBtn);

    auto* fusedHeader = new QHBoxLayout;
    fusedHeader->addWidget(fusedBtn);
    fusedHeader->addWidget(saveResultsBtn);

    auto* visLayout = new QVBoxLayout;
    visLayout->addLayout(visHeader);
    visLayout->addWidget(visLabel);

    auto* irLayout = new QVBoxLayout;
    irLayout->addLayout(irHeader);
    irLayout->addWidget(irLabel);

    auto* fusedLayout = new QVBoxLayout;
    fusedLayout->addLayout(fusedHeader);
    fusedLayout->addWidget(resultTab);

    auto* imageLayout = new QHBoxLayout;
    imageLayout->addLayout(visLayout, 1);
    imageLayout->addLayout(irLayout, 1);
    imageLayout->addLayout(fusedLayout, 1);
    imageLayout->setSpacing(12);

    fusionPage = new QWidget;
    auto* fusionLayout = new QVBoxLayout(fusionPage);
    fusionLayout->addLayout(percentileLoRow);
    fusionLayout->addLayout(percentileHiRow);
    fusionLayout->addLayout(imageLayout);
    fusionLayout->addWidget(statusLabel);
    fusionLayout->addWidget(log);
    fusionLayout->setSpacing(10);
    fusionLayout->setContentsMargins(12, 12, 12, 12);

    // =========================
    // Comparison / Triage page
    // =========================
    triageLoadBtn = new QPushButton("Load IR");
    triageRunBtn = new QPushButton("Enhance");
    triageSaveBtn = new QPushButton("Save Triage");
    triageViewBtn = new QPushButton("Show differences");

    compareInputTab = makeImageLabel("Input");
    compareRGFTab = makeImageLabel("RGF");
    compareMSGFTab = makeImageLabel("MSGF");
    compareHybridTab = makeImageLabel("Proposed");

    triageLog = new QPlainTextEdit;
    triageLog->setReadOnly(true);
    triageLog->setMaximumBlockCount(300);
    triageLog->setMinimumHeight(170);
    triageLog->setStyleSheet(
        "QPlainTextEdit {"
        "background-color: #111111;"
        "color: #d8d8d8;"
        "border: 1px solid #555;"
        "font-family: Consolas, monospace;"
        "font-size: 12px;"
        "}"
    );

    auto* triageButtons = new QHBoxLayout;
    triageButtons->addWidget(triageLoadBtn);
    triageButtons->addWidget(triageRunBtn);
    triageButtons->addWidget(triageSaveBtn);
    triageButtons->addWidget(triageViewBtn);
    triageButtons->addStretch();

    auto* triageGrid = new QGridLayout;
    triageGrid->addWidget(compareInputTab, 0, 0);
    triageGrid->addWidget(compareRGFTab, 0, 1);
    triageGrid->addWidget(compareMSGFTab, 0, 2);
    triageGrid->addWidget(compareHybridTab, 0, 3);
    triageGrid->setSpacing(12);

    comparePage = new QWidget;
    auto* compareLayout = new QVBoxLayout(comparePage);
    compareLayout->addLayout(triageButtons);
    compareLayout->addLayout(triageGrid, 1);
    compareLayout->addWidget(triageLog);
    compareLayout->setSpacing(10);
    compareLayout->setContentsMargins(12, 12, 12, 12);

    // =========================
    // Main tab
    // =========================
    mainTab = new QTabWidget;
    mainTab->addTab(fusionPage, "Fusion");
    mainTab->addTab(comparePage, "Comparison");

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(mainTab);
    mainLayout->setContentsMargins(6, 6, 6, 6);

    // =========================
    // Connections
    // =========================
    connect(visBtn, &QPushButton::clicked, this, &MainWindow::loadVisImg);
    connect(irBtn, &QPushButton::clicked, this, &MainWindow::loadIrImg);
    connect(fusedBtn, &QPushButton::clicked, this, &MainWindow::fuse);
    connect(clearVisBtn, &QPushButton::clicked, this, &MainWindow::clearVis);
    connect(clearIrBtn, &QPushButton::clicked, this, &MainWindow::clearIr);
    connect(saveResultsBtn, &QPushButton::clicked, this, &MainWindow::saveResults);

    connect(triageLoadBtn, &QPushButton::clicked, this, &MainWindow::loadIrImg);
    connect(triageRunBtn, &QPushButton::clicked, this, &MainWindow::runTriage);
    connect(triageSaveBtn, &QPushButton::clicked, this, &MainWindow::saveTriage);
    connect(triageViewBtn, &QPushButton::clicked, this, [this] {
        triageDiff = !triageDiff;
        triageViewBtn->setText(triageDiff ? "Show Normal" : "Show Differences");
        updateTriage();
        appendTriageLog(triageDiff ? "Different View ON" : "Different View OFF");
    });

    connect(percentileLoSlider, &QSlider::valueChanged, this, [this](const int value) {
        if (value >= percentileHiSlider->value()) {
            percentileLoSlider->blockSignals(true);
            percentileLoSlider->setValue(percentileHiSlider->value() - 1);
            percentileLoSlider->blockSignals(false);
        }
        percentileLoLabel->setText(QString("Low: %1.0").arg(percentileLoSlider->value()));
    });

    connect(percentileHiSlider, &QSlider::valueChanged, this, [this](const int value) {
        if (value <= percentileLoSlider->value()) {
            percentileHiSlider->blockSignals(true);
            percentileHiSlider->setValue(percentileLoSlider->value() + 1);
            percentileHiSlider->blockSignals(false);
        }
        percentileHiLabel->setText(QString("High: %1.0").arg(percentileHiSlider->value()));
    });

    updateStatus();
    appendLog("Application started.");
    appendTriageLog("Comparison tab ready.");
}

void MainWindow::appendLog(const QString& msg) const {
    const QString timestamp = QDateTime::currentDateTime().toString("HH:mm:ss");
    log->appendPlainText(QString("[%1] %2").arg(timestamp, msg));
}

void MainWindow::appendTriageLog(const QString& msg) const {
    const QString timestamp = QDateTime::currentDateTime().toString("HH:mm:ss");
    triageLog->appendPlainText(QString("[%1] %2").arg(timestamp, msg));
}

void MainWindow::updateStatus() const {
    const int visIndex = extractIndex(visPath, "VIS");
    const int irIndex = extractIndex(irPath, "IR");

    const QString visText = (visIndex >= 0) ? QString::number(visIndex) : "-";
    const QString irText = (irIndex >= 0) ? QString::number(irIndex) : "-";

    QString state = "Not ready";

    if (!visImg.empty() && !irImg.empty()) {
        if (visIndex < 0 || irIndex < 0) {
            state = "Invalid filename";
        } else if (visIndex != irIndex) {
            state = "Pair mismatch";
        } else {
            state = "Ready to fuse";
        }
    }

    statusLabel->setText(QString("VIS: %1 | IR: %2 | %3").arg(visText, irText, state));
}

void MainWindow::saveResults() {
    if (!hasResult) {
        QMessageBox::warning(this, "Save", "No fusion result available to save.");
        appendLog("Save blocked: no result available.");
        return;
    }

    const QString tabName = resultTab->tabText(resultTab->currentIndex());

    cv::Mat imageToSave;
    QString defaultName;

    QMap<QString, std::function<void()>> actions = {
        {
            "Fused", [&] {
                imageToSave = final.fused_image;
                defaultName = "fused";
            }
        },
        {
            "Thermal", [&] {
                auto it = final.dictionary.find("thermal8");
                if (it != final.dictionary.end()) {
                    imageToSave = Displays(it->second);
                    defaultName = "thermal8";
                }
            }
        },
        {
            "Base", [&] {
                auto it = final.dictionary.find("B_enh");
                if (it != final.dictionary.end()) {
                    imageToSave = Displays(it->second);
                    defaultName = "enhanced_base";
                }
            }
        },
        {
            "Thermal+", [&] {
                auto it = final.dictionary.find("L_T");
                if (it != final.dictionary.end()) {
                    imageToSave = Displays(it->second);
                    defaultName = "enhanced_thermal";
                }
            }
        },
        {
            "Details", [&] {
                auto itFine = final.dictionary.find("D_fine_enh");
                auto itStruct = final.dictionary.find("D_struct_enh");
                if (itFine != final.dictionary.end() && itStruct != final.dictionary.end()) {
                    imageToSave = Displays(itFine->second + itStruct->second);
                    defaultName = "enhanced_details";
                }
            }
        }
    };

    auto action = actions.find(tabName);
    if (action != actions.end()) {
        action.value()();
    }

    if (imageToSave.empty()) {
        QMessageBox::warning(this, "Save", "Current tab has no image to save.");
        appendLog(QString("Save blocked: tab '%1' has no image.").arg(tabName));
        return;
    }

    const int visIndex = extractIndex(visPath, "VIS");
    const int irIndex = extractIndex(irPath, "IR");
    QString suggestedName = defaultName;
    if (visIndex >= 0 && irIndex >= 0 && visIndex == irIndex) {
        suggestedName += QString("_%1").arg(visIndex);
    }

    const QString fileName = QFileDialog::getSaveFileName(
        this,
        "Save Current Tab",
        QString("F:/RGB_Thermal_Fusion__C++/output/%1.png").arg(suggestedName),
        "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap (*.bmp)"
    );

    if (fileName.isEmpty()) {
        return;
    }

    if (!cv::imwrite(fileName.toStdString(), imageToSave)) {
        QMessageBox::critical(this, "Save Error", "Failed to save image.");
        appendLog(QString("Failed to save '%1'.").arg(fileName));
        return;
    }

    appendLog(QString("Saved tab '%1' to %2").arg(tabName, fileName));
}

void MainWindow::updateTriage() const {
    if (triageInputImg.empty()) {
        compareInputTab->clear();
        compareInputTab->setText("Input");

        compareRGFTab->clear();
        compareRGFTab->setText("RGF");

        compareMSGFTab->clear();
        compareMSGFTab->setText("MSGF");

        compareHybridTab->clear();
        compareHybridTab->setText("Proposed");
        return;
    }
    updateLabel(compareInputTab, triageInputImg);
    updateLabel(compareHybridTab, triageHybrid);

    if (!triageDiff) {
        updateLabel(compareRGFTab, triageRGF);
        updateLabel(compareMSGFTab, triageMSGF);
    } else {
        if (!triageDiffRGF.empty()) {
            updateLabel(compareRGFTab, triageDiffRGF);
        } else {
            compareRGFTab->clear();
            compareRGFTab->setText("Diff vs RGF");
        }

        if (!triageDiffMSGF.empty()) {
            updateLabel(compareMSGFTab, triageDiffMSGF);
        } else {
            compareMSGFTab->clear();
            compareMSGFTab->setText("Diff vs MSGF");
        }
    }
}

void MainWindow::saveTriage() {
    if (triageInputImg.empty() || triageRGF.empty() || triageMSGF.empty() || triageHybrid.empty()) {
        QMessageBox::warning(this, "Save Triage", "No comparison results available to save.");
        appendTriageLog("Save blocked: comparison results are empty.");
        return;
    }

    const QString fileName = QFileDialog::getSaveFileName(
        this,
        "Save Triage Panel",
        "F:/RGB_Thermal_Fusion__C++/output/triage_panel.png",
        "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap (*.bmp)"
    );

    if (fileName.isEmpty()) {
        return;
    }

    try {
        triage::write_panel(fileName.toStdString(), {
            {"Input", triageInputImg},
            {"RGF", triageRGF},
            {"MSGF", triageMSGF},
            {"Proposed", triageHybrid}
        });
        appendTriageLog(QString("Saved comparison panel to %1").arg(fileName));
    }
    catch (const std::exception& e) {
        QMessageBox::critical(this, "Save Triage Error", e.what());
        appendTriageLog(QString("Failed to save comparison panel: %1").arg(e.what()));
    }
}

void MainWindow::updateLabel(QLabel* label, const cv::Mat& img) {
    const QImage qimg = matToQImage(img);
    if (qimg.isNull()) {
        label->clear();
        label->setText("Empty");
        return;
    }

    const QPixmap pixmap = QPixmap::fromImage(qimg);
    if (pixmap.isNull()) {
        label->clear();
        label->setText("Pixmap failed");
        return;
    }

    label->setPixmap(
        pixmap.scaled(
            label->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        )
    );
}

cv::Mat MainWindow::Displays(const cv::Mat& img) {
    if (img.empty()) {
        return {};
    }

    if (img.type() == CV_8UC1 || img.type() == CV_8UC3 || img.type() == CV_8UC4) {
        return img;
    }

    cv::Mat src32f;
    img.convertTo(src32f, CV_32F);

    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(src32f, &minVal, &maxVal);

    if (maxVal - minVal < 1e-6) {
        return cv::Mat::zeros(img.size(), CV_8UC1);
    }

    const cv::Mat normalized = (src32f - static_cast<float>(minVal)) /
                               static_cast<float>(maxVal - minVal);

    cv::Mat out;
    normalized.convertTo(out, CV_8U, 255.0);
    return out;
}

void MainWindow::clearResults() {
    fusedImg.release();
    hasResult = false;

    fusedTab->clear();
    fusedTab->setText("Fused");

    irTab->clear();
    irTab->setText("Thermal 8-bit");

    baseTab->clear();
    baseTab->setText("Enhanced Base");

    enhancedTab->clear();
    enhancedTab->setText("Enhanced Thermal");

    detailsTab->clear();
    detailsTab->setText("Enhanced Details");
}

void MainWindow::clearTriage() {
    triageInputImg.release();
    triageRGF.release();
    triageMSGF.release();
    triageHybrid.release();
    triageDiffRGF.release();
    triageDiffMSGF.release();

    triageDiff = false;
    if (triageViewBtn) {
        triageViewBtn->setText("Show Differences");
    }

    compareInputTab->clear();
    compareInputTab->setText("Input");

    compareRGFTab->clear();
    compareRGFTab->setText("RGF");

    compareMSGFTab->clear();
    compareMSGFTab->setText("MSGF");

    compareHybridTab->clear();
    compareHybridTab->setText("Proposed");
}

void MainWindow::clearVis() {
    visImg.release();
    visPath.clear();

    visLabel->clear();
    visLabel->setText("Visible image");

    clearResults();
    updateStatus();
    appendLog("Cleared VIS image.");
}

void MainWindow::clearIr() {
    irImg.release();
    irPath.clear();

    irLabel->clear();
    irLabel->setText("Thermal image");

    clearResults();
    clearTriage();
    updateStatus();
    appendLog("Cleared IR image.");
    appendTriageLog("Cleared IR image.");
}

void MainWindow::loadVisImg() {
    const QString fileName = QFileDialog::getOpenFileName(
        this,
        "Open VIS Image",
        kDataDir,
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    );

    if (fileName.isEmpty()) {
        return;
    }

    const int visIndex = extractIndex(fileName, "VIS");
    if (visIndex < 0) {
        QMessageBox::warning(this, "Invalid filename", "Visible image must be named like VIS1, VIS2, VIS3, ...");
        appendLog(QString("Rejected VIS file: %1").arg(fileOnly(fileName)));
        return;
    }

    cv::Mat loadedVis;
    if (!loadImageChecked(fileName, cv::IMREAD_COLOR, loadedVis)) {
        QMessageBox::critical(this, "Error", "Failed to load visible image.");
        appendLog(QString("Failed to load VIS image: %1").arg(fileOnly(fileName)));
        return;
    }

    visImg = loadedVis;
    visPath = fileName;
    updateLabel(visLabel, visImg);
    appendLog(QString("Loaded VIS%1: %2").arg(visIndex).arg(fileOnly(fileName)));

    const QString pairedIrPath = findCounterpartByIndex(fileName, "IR");
    if (!pairedIrPath.isEmpty()) {
        if (cv::Mat loadedIr; loadImageChecked(pairedIrPath, cv::IMREAD_COLOR, loadedIr)) {
            irImg = loadedIr;
            irPath = pairedIrPath;
            updateLabel(irLabel, irImg);
            appendLog(QString("Auto-loaded IR%1: %2").arg(visIndex).arg(fileOnly(pairedIrPath)));
        } else {
            appendLog(QString("Found paired IR file but failed to load: %1").arg(fileOnly(pairedIrPath)));
        }
    } else {
        irImg.release();
        irPath.clear();
        irLabel->clear();
        irLabel->setText("Thermal image");
        appendLog(QString("No paired IR%1 found in same folder.").arg(visIndex));
    }

    clearResults();
    updateStatus();
}

void MainWindow::loadIrImg() {
    const QString fileName = QFileDialog::getOpenFileName(
        this,
        "Open IR Image",
        kDataDir,
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    );

    if (fileName.isEmpty()) {
        return;
    }

    const int irIndex = extractIndex(fileName, "IR");
    if (irIndex < 0) {
        QMessageBox::warning(this, "Invalid filename", "Thermal image must be named like IR1, IR2, IR3, ...");
        appendLog(QString("Rejected IR file: %1").arg(fileOnly(fileName)));
        appendTriageLog(QString("Rejected IR file: %1").arg(fileOnly(fileName)));
        return;
    }

    cv::Mat loadedIr;
    if (!loadImageChecked(fileName, cv::IMREAD_COLOR, loadedIr)) {
        QMessageBox::critical(this, "Error", "Failed to load thermal image.");
        appendLog(QString("Failed to load IR image: %1").arg(fileOnly(fileName)));
        appendTriageLog(QString("Failed to load IR image: %1").arg(fileOnly(fileName)));
        return;
    }

    irImg = loadedIr;
    irPath = fileName;
    updateLabel(irLabel, irImg);
    appendLog(QString("Loaded IR%1: %2").arg(irIndex).arg(fileOnly(fileName)));
    appendTriageLog(QString("Loaded IR%1: %2").arg(irIndex).arg(fileOnly(fileName)));

    // also show it immediately on comparison page
    triageInputImg = Displays(irImg);
    triageRGF.release();
    triageMSGF.release();
    triageHybrid.release();
    triageDiffRGF.release();
    triageDiffMSGF.release();

    triageDiff = false;
    if (triageViewBtn) {
        triageViewBtn->setText("Show Differences");
    }
    updateTriage();

    const QString pairedVisPath = findCounterpartByIndex(fileName, "VIS");
    if (!pairedVisPath.isEmpty()) {
        if (cv::Mat loadedVis; loadImageChecked(pairedVisPath, cv::IMREAD_COLOR, loadedVis)) {
            visImg = loadedVis;
            visPath = pairedVisPath;
            updateLabel(visLabel, visImg);
            appendLog(QString("Auto-loaded VIS%1: %2").arg(irIndex).arg(fileOnly(pairedVisPath)));
        } else {
            appendLog(QString("Found paired VIS file but failed to load: %1").arg(fileOnly(pairedVisPath)));
        }
    } else {
        visImg.release();
        visPath.clear();
        visLabel->clear();
        visLabel->setText("Visible image");
        appendLog(QString("No paired VIS%1 found in same folder.").arg(irIndex));
    }

    clearResults();
    updateStatus();
}

void MainWindow::updateTabs() {
    if (!hasResult) {
        clearResults();
        return;
    }

    updateLabel(fusedTab, final.fused_image);

    if (final.dictionary.count("thermal8")) {
        updateLabel(irTab, Displays(final.dictionary.at("thermal8")));
    }

    if (final.dictionary.count("B_enh")) {
        updateLabel(baseTab, Displays(final.dictionary.at("B_enh")));
    }

    if (final.dictionary.count("L_T")) {
        updateLabel(enhancedTab, Displays(final.dictionary.at("L_T")));
    }

    if (final.dictionary.count("D_fine_enh") && final.dictionary.count("D_struct_enh")) {
        const cv::Mat detail = final.dictionary.at("D_fine_enh") + final.dictionary.at("D_struct_enh");
        updateLabel(detailsTab, Displays(detail));
    }
}

void MainWindow::fuse() {
    if (visImg.empty() || irImg.empty()) {
        QMessageBox::critical(this, "Error", "Missing one or two required images!");
        appendLog("Fusion blocked: missing VIS or IR image.");
        return;
    }

    const int visIndex = extractIndex(visPath, "VIS");
    const int irIndex = extractIndex(irPath, "IR");

    if (visIndex < 0 || irIndex < 0) {
        QMessageBox::critical(this, "Invalid filename", "Fusion requires files named like VIS<number> and IR<number>.");
        appendLog("Fusion blocked: invalid VIS/IR filename pattern.");
        updateStatus();
        return;
    }

    if (visIndex != irIndex) {
        QMessageBox::critical(
            this,
            "Pair mismatch",
            QString("VIS%1 and IR%2 do not match. Please load the same numbered pair.")
                .arg(visIndex)
                .arg(irIndex)
        );
        appendLog(QString("Fusion blocked: VIS%1 does not match IR%2").arg(visIndex).arg(irIndex));
        updateStatus();
        return;
    }

    const auto percentileLo = static_cast<float>(percentileLoSlider->value());
    const auto percentileHi = static_cast<float>(percentileHiSlider->value());

    appendLog(QString("Fusion started for pair %1 | percentile_lo=%2 | percentile_hi=%3")
              .arg(visIndex)
              .arg(percentileLo, 0, 'f', 1)
              .arg(percentileHi, 0, 'f', 1));

    QElapsedTimer timer;
    timer.start();

    try {
        te::fusionParameters params{};
        params.percentile_lo = percentileLo;
        params.percentile_hi = percentileHi;

        te::fusionResult res = te::rgb_fusion(visImg, irImg, nullptr, params);

        fusedImg = res.fused_image;
        final = std::move(res);
        hasResult = true;
    }
    catch (const std::exception& e) {
        QMessageBox::critical(this, "Fusing error", e.what());
        appendLog(QString("Fusion failed: %1").arg(e.what()));
        return;
    }

    if (fusedImg.empty()) {
        QMessageBox::critical(this, "Error", "Empty fusion return!");
        appendLog("Fusion failed: empty output image.");
        return;
    }

    updateTabs();
    resultTab->setCurrentIndex(0);
    updateStatus();

    const qint64 elapsedMs = timer.elapsed();
    appendLog(QString("Fusion finished for pair %1 in %2 ms.").arg(visIndex).arg(elapsedMs));
}

void MainWindow::runTriage() {
    if (irImg.empty()) {
        QMessageBox::warning(this, "Comparison", "Please load a thermal image first.");
        appendTriageLog("Comparison blocked: no IR image loaded.");
        return;
    }

    appendTriageLog("Comparison started.");

    try {
        QElapsedTimer totalTimer;
        totalTimer.start();

        cv::Mat irGray;
        if (irImg.channels() == 3) {
            cv::cvtColor(irImg, irGray, cv::COLOR_BGR2GRAY);
        } else {
            irGray = irImg.clone();
        }

        // -------- RGF --------
        QElapsedTimer rgfTimer;
        rgfTimer.start();

        cv::Mat rgf_thermal8, rgf_base, rgf_detail;
        ref_rgf::rgf_thermal(irGray, rgf_thermal8, rgf_base, rgf_detail,
                             4, 6, 1e-3f, 1.5f, 1.0f, 99.0f);

        const qint64 rgfMs = rgfTimer.elapsed();

        cv::Mat rgf_u8;
        cv::Mat rgf_base_clipped;
        cv::min(cv::max(rgf_base, 0.0f), 1.0f, rgf_base_clipped);
        rgf_base_clipped.convertTo(rgf_u8, CV_8U, 255.0);

        // -------- Proposed --------
        QElapsedTimer hybridTimer;
        hybridTimer.start();

        cv::Mat fake_rgb;
        cv::cvtColor(rgf_thermal8, fake_rgb, cv::COLOR_GRAY2BGR);

        te::fusionParameters pmt{};
        auto hybrid_result = te::rgb_fusion(fake_rgb, irGray, nullptr, pmt);

        const qint64 hybridMs = hybridTimer.elapsed();

        cv::Mat hybrid_L = hybrid_result.dictionary["L_T"];
        cv::Mat hybrid_clipped, hybrid_u8;
        cv::min(cv::max(hybrid_L, 0.0f), 1.0f, hybrid_clipped);
        hybrid_clipped.convertTo(hybrid_u8, CV_8U, 255.0);

        // -------- MSGF --------
        QElapsedTimer msgfTimer;
        msgfTimer.start();

        cv::Mat msgf_thermal8, msgf_base, msgf_base_equal, msgf_mix;
        std::vector<cv::Mat> msgf_details;
        ref_msgf::msgf_thermal(irGray, msgf_thermal8, msgf_base, msgf_details, msgf_base_equal, msgf_mix,
                               {3, 8, 16, 32}, 1e-3f, 2.0, 8, {0.18f, 0.12f, 0.06f, 0.03f});

        const qint64 msgfMs = msgfTimer.elapsed();

        cv::Mat msgf_u8;
        cv::Mat msgf_clipped;
        cv::min(cv::max(msgf_mix, 0.0f), 1.0f, msgf_clipped);
        msgf_clipped.convertTo(msgf_u8, CV_8U, 255.0);

        // -------- Cache + display --------
        triageInputImg = Displays(irGray);
        triageRGF = Displays(rgf_u8);
        triageMSGF = Displays(msgf_u8);
        triageHybrid = Displays(hybrid_u8);
        cv::absdiff(triageHybrid, triageRGF, triageDiffRGF);
        cv::absdiff(triageHybrid, triageMSGF, triageDiffMSGF);
        cv::normalize(triageDiffRGF, triageDiffRGF, 0, 255, cv::NORM_MINMAX);
        cv::normalize(triageDiffMSGF, triageDiffMSGF, 0, 255, cv::NORM_MINMAX);

        updateTriage();

        // -------- Metrics --------
        const auto m_input = triage::metrics(triageInputImg);
        const auto m_rgf = triage::metrics(triageRGF);
        const auto m_msgf = triage::metrics(triageMSGF);
        const auto m_hybrid = triage::metrics(triageHybrid);

        triageLog->clear();
        appendTriageLog(metricsToQString("Input   ", m_input));
        appendTriageLog(metricsToQString("RGF     ", m_rgf));
        appendTriageLog(metricsToQString("MSGF    ", m_msgf));
        appendTriageLog(metricsToQString("Proposed", m_hybrid));
        appendTriageLog(QString("Time | RGF=%1 ms | MSGF=%2 ms | Proposed=%3 ms | Total=%4 ms")
                        .arg(rgfMs)
                        .arg(msgfMs)
                        .arg(hybridMs)
                        .arg(totalTimer.elapsed()));

        appendTriageLog("Comparison finished.");
    }
    catch (const std::exception& e) {
        QMessageBox::critical(this, "Comparison Error", e.what());
        appendTriageLog(QString("Comparison failed: %1").arg(e.what()));
    }
}