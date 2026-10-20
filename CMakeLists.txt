cmake_minimum_required(VERSION 3.20)
project(RGB_Thermal_Fusion__C__ LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

include(CheckIPOSupported)

find_package(OpenCV REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# =========================================================
# IPO/LTO: enable safely instead of hard-forcing -flto
# =========================================================
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    message(STATUS "IPO/LTO not supported: ${ipo_error}")
endif()

# =========================================================
# Core library: current hybrid C++ module
# =========================================================
add_library(te_rgf_rgb STATIC
        src/te_rgf_rgb.cpp
        include/te_rgf_rgb.hpp
)

target_include_directories(te_rgf_rgb
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(te_rgf_rgb
        PUBLIC
        ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(te_rgf_rgb PRIVATE
            $<$<CONFIG:Release>:/O2>
    )
else()
    target_compile_options(te_rgf_rgb PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native>
    )
endif()

# =========================================================
# Reference ports for baseline comparison
# =========================================================
add_library(thermal_references STATIC
        src/rgf.cpp
        src/msgf.cpp
        include/rgf.hpp
        include/msgf.hpp
)

target_include_directories(thermal_references
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(thermal_references
        PUBLIC
        ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(thermal_references PRIVATE
            $<$<CONFIG:Release>:/O2>
    )
else()
    target_compile_options(thermal_references PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native>
    )
endif()

# =========================================================
# Evaluation utilities and metrics
# =========================================================
add_library(triage_metrics STATIC
        src/triage_metrics.cpp
        include/triage_metrics.hpp
)

target_include_directories(triage_metrics
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(triage_metrics
        PUBLIC
        ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(triage_metrics PRIVATE
            $<$<CONFIG:Release>:/O2>
    )
else()
    target_compile_options(triage_metrics PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native>
    )
endif()

# =========================================================
# CLI app
# =========================================================
add_executable(rgf_rgb_app
        apps/cli/main.cpp
)

target_link_libraries(rgf_rgb_app
        PRIVATE
        te_rgf_rgb
)

# =========================================================
# Qt GUI app
# =========================================================
add_executable(rgf_rgb_qt
        apps/gui_qt/main.cpp
        apps/gui_qt/MainWindow.hpp
        apps/gui_qt/MainWindow.cpp
        src/triage_metrics.cpp
        src/rgf.cpp
        src/msgf.cpp
)

target_link_libraries(rgf_rgb_qt
        PRIVATE
        te_rgf_rgb
        Qt6::Widgets
)

# =========================================================
# Triage runner for 4-metric evaluation
# =========================================================
add_executable(run_triage
        src/run_triage.cpp
)

target_link_libraries(run_triage
        PRIVATE
        te_rgf_rgb
        thermal_references
        triage_metrics
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
        include/te_rgf_rgb.hpp
        include/rgf.hpp
        include/msgf.hpp
        include/triage_metrics.hpp
        src/te_rgf_rgb.cpp
        src/rgf.cpp
        src/msgf.cpp
        src/triage_metrics.cpp
        src/run_triage.cpp
        apps/cli/main.cpp
        apps/gui_qt/main.cpp
        apps/gui_qt/MainWindow.hpp
        apps/gui_qt/MainWindow.cpp
)

add_executable(evaluate_legacy_thermal
        tools/evaluate_legacy_thermal.cpp
)

target_link_libraries(evaluate_legacy_thermal
        PRIVATE
        te_rgf_rgb
        triage_metrics
)

add_executable(export_legacy_stages
        tools/export_legacy_stages.cpp
)

target_link_libraries(export_legacy_stages
        PRIVATE
        te_rgf_rgb
)

add_executable(compare_legacy_clahe_grid
        tools/compare_legacy_clahe_grid.cpp
)

target_link_libraries(compare_legacy_clahe_grid
        PRIVATE
        te_rgf_rgb
        triage_metrics
)

add_executable(compare_legacy_clahe_grid40
        tools/compare_legacy_clahe_grid40.cpp
)

target_link_libraries(compare_legacy_clahe_grid40
        PRIVATE
        te_rgf_rgb
        triage_metrics
)

add_executable(compare_legacy_fine_gain40
        tools/compare_legacy_fine_gain40.cpp
)

target_link_libraries(compare_legacy_fine_gain40
        PRIVATE
        te_rgf_rgb
        triage_metrics
)

set(QART_ACCEPTED_CPP_ROOT
        "F:/qtRoadmap/Visual_Thermal_Concept"
        CACHE PATH
        "Checkout containing accepted N5 source"
)

set(QART_PROBE_N5_SOURCE
        "${QART_ACCEPTED_CPP_ROOT}/src/core/enhancement/n5_thermal_enhancer.cpp"
)

if(NOT EXISTS "${QART_PROBE_N5_SOURCE}")
    message(FATAL_ERROR "Accepted N5 source not found")
endif()

add_executable(run_legacy_final_probe
        tools/run_legacy_final_probe.cpp
        "${QART_PROBE_N5_SOURCE}"
)

target_include_directories(run_legacy_final_probe
        PRIVATE
        "${QART_ACCEPTED_CPP_ROOT}/src"
)

target_link_libraries(run_legacy_final_probe
        PRIVATE
        te_rgf_rgb
        ${OpenCV_LIBS}
)

if(MSVC)
    set_property(SOURCE "${QART_PROBE_N5_SOURCE}"
            APPEND PROPERTY COMPILE_OPTIONS /fp:strict)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set_property(SOURCE "${QART_PROBE_N5_SOURCE}"
            APPEND PROPERTY COMPILE_OPTIONS
            -ffp-contract=off -fno-fast-math)
endif()

set(
        QART_ACCEPTED_CPP_ROOT
        "F:/qtRoadmap/Visual_Thermal_Concept"
        CACHE PATH
        "Accepted QART C++ repository"
)

set(
        QART_ACCEPTED_N5_SOURCE
        "${QART_ACCEPTED_CPP_ROOT}/src/core/enhancement/n5_thermal_enhancer.cpp"
)

if(NOT EXISTS "${QART_ACCEPTED_N5_SOURCE}")
    message(
            FATAL_ERROR
            "Accepted N5 source not found: ${QART_ACCEPTED_N5_SOURCE}"
    )
endif()

add_executable(
        evaluate_selective_n5
        tools/evaluate_selective_n5.cpp
        tools/selective_n5_candidate.cpp
        src/te_rgf_rgb.cpp
        "${QART_ACCEPTED_N5_SOURCE}"
)

target_sources(
        evaluate_selective_n5
        PRIVATE
        tools/legacy_thermal_candidate.cpp
)

target_compile_features(
        evaluate_selective_n5
        PRIVATE
        cxx_std_17
)

target_include_directories(
        evaluate_selective_n5
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${QART_ACCEPTED_CPP_ROOT}/src/core/enhancement"
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
        evaluate_selective_n5
        PRIVATE
        ${OpenCV_LIBS}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(
            evaluate_selective_n5
            PRIVATE
            -ffp-contract=off
            -fno-fast-math
    )
elseif(MSVC)
    target_compile_options(
            evaluate_selective_n5
            PRIVATE
            /fp:strict
    )
endif()

add_executable(
        test_legacy_thermal_contract
        tools/test_legacy_thermal_contract.cpp
        tools/legacy_thermal_candidate.cpp
)

target_compile_features(
        test_legacy_thermal_contract
        PRIVATE
        cxx_std_17
)

target_include_directories(
        test_legacy_thermal_contract
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tools"
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
        test_legacy_thermal_contract
        PRIVATE
        ${OpenCV_LIBS}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(
            test_legacy_thermal_contract
            PRIVATE
            -ffp-contract=off
            -fno-fast-math
    )
elseif(MSVC)
    target_compile_options(
            test_legacy_thermal_contract
            PRIVATE
            /fp:strict
    )
endif()