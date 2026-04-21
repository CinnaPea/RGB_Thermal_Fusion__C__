# RGB Thermal Fusion (C++)

A C++ project for enhancing thermal images and performing RGB–thermal fusion using traditional image processing techniques.  
Designed for real-time feasibility with a Qt-based visualization interface.

## Overview

This project focuses on improving the visual quality of thermal images, which typically suffer from:
- Low contrast
- High noise
- Weak structural details

The proposed method applies:
- Multi-scale decomposition (RGF-based)
- Selective detail enhancement (entropy + contrast)
- CLAHE-based base layer enhancement
- Optional RGB–thermal fusion using luminance (Y channel)

## Features

- Thermal image enhancement pipeline
- Multi-scale decomposition (RGF, MSGF-inspired)
- Adaptive detail enhancement
- Luminance-based RGB–thermal fusion (YCrCb)
- Qt GUI for visualization
- Metric evaluation (Entropy, Sobel, Laplacian, RMS contrast)

## Tech Stack

- C++
- OpenCV
- Qt (Qt6 Widgets)
- CMake

## 📁 Project Structure
