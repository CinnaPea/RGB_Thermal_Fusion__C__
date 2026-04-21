# RGB–Thermal Image Enhancement & Fusion (C++ / Qt)

## Overview

This project presents a complete implementation of a **thermal image enhancement pipeline** built using traditional image processing techniques, along with an optional **RGB–thermal fusion module** for improved visual perception in multi-sensor systems.

The primary objective is to improve the **observability of thermal images**, which are often degraded by:
- low contrast
- high noise levels
- lack of fine structural detail

Instead of relying on deep learning, this work focuses on **interpretable, efficient, and deployable methods**, making it suitable for real-world systems with limited computational resources.


## Key Features

### Thermal Image Enhancement
- Multi-scale decomposition using **Rolling Guidance Filter (RGF)**
- Separation into base layer and detail layers
- Selective detail enhancement based on:
  - local entropy
  - local contrast
- Controlled contrast improvement using **CLAHE**

### RGB–Thermal Fusion (Optional Module)
- Fusion performed in **luminance (Y) channel**
- Preserves original RGB color information
- Enhances structural visibility using thermal data

### Qt-based GUI
- Interactive visualization of:
  - input images
  - intermediate layers (base, detail)
  - final enhanced output
- Comparison tabs (RGF / MSGF / Proposed method)
- Triage panel for metric evaluation

### Evaluation Metrics (No-reference)
- Entropy
- Sobel energy
- Laplacian variance
- RMS contrast


## Design Philosophy

This project intentionally avoids deep learning approaches and instead emphasizes:

- **Transparency** – every step is explainable
- **Control** – fine-grained parameter tuning
- **Efficiency** – suitable for CPU-based systems
- **Deployability** – realistic for embedded or constrained environments


## Technologies Used

- **C++**
- **OpenCV**
- **Qt (Widgets)**
- **CMake**


## Instructions

### Requirements
- CMake ≥ 3.x
- OpenCV
- Qt (Widgets module)

### Documentation
For a deeper understanding, please refer to the full report in /docs

### Author Notes

This project was developed as part of an academic research effort focused on practical image processing methods for thermal imaging systems.
The goal was not just to achieve results, but to build a balanced, explainable, and implementable pipeline
