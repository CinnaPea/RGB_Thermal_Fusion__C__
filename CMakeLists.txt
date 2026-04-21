#cmake_minimum_required(VERSION 3.20)
#project(RGB_Thermal_Fusion__C__ LANGUAGES C CXX)
#
#set(CMAKE_C_STANDARD 11)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
#set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
#set(CMAKE_AUTORCC ON)
#
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
#set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
#
#
#find_package(OpenCV REQUIRED)
#find_package(Qt6 REQUIRED COMPONENTS Widgets)
#
#add_library(te_rgf_rgb STATIC
#    src/te_rgf_rgb.cpp
#)
#target_include_directories(te_rgf_rgb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_link_libraries(te_rgf_rgb PUBLIC ${OpenCV_LIBS})
#
#add_executable(rgf_rgb_app
#        apps/cli/main.cpp
#        include/te_rgf_rgb.hpp
#)
#target_link_libraries(rgf_rgb_app PRIVATE te_rgf_rgb)
#
#add_executable(rgf_rgb_qt
#        apps/gui_qt/main.cpp
#        apps/gui_qt/MainWindow.hpp
#        apps/gui_qt/MainWindow.cpp
#        include/te_rgf_rgb.hpp
#)
#target_link_libraries(rgf_rgb_qt PRIVATE te_rgf_rgb Qt6::Widgets)

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
