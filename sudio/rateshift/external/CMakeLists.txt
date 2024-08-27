cmake_minimum_required(VERSION 3.15)  # FetchContent requires CMake 3.14+

include(FetchContent)

# Check if pybind11 directory exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pybind11")
    message(STATUS "Using existing pybind11 in ${CMAKE_CURRENT_SOURCE_DIR}/pybind11")
    add_subdirectory(pybind11)
else()
    message(STATUS "cant find pybind11 in ${CMAKE_CURRENT_SOURCE_DIR}/pybind11 Fetching pybind11.................")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG 5b0a6fc2017fcc176545afe3e09c9f9885283242 # 2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# libsamplerate
set(BUILD_TESTING OFF CACHE BOOL "Disable libsamplerate test build")
set(LIBSAMPLERATE_VERSION 0.2.2 CACHE STRING "libsamplerate version")

# Check if libsamplerate directory exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate")
    message(STATUS "Using existing libsamplerate in ${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate")
    add_subdirectory(libsamplerate)
else()
    message(STATUS "cant find libsamplerate in ${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate Fetching libsamplerate...................")
    FetchContent_Declare(
        libsamplerate
        GIT_REPOSITORY https://github.com/libsndfile/libsamplerate
        GIT_TAG c96f5e3de9c4488f4e6c97f59f5245f22fda22f7 # 0.2.2
    )
    FetchContent_MakeAvailable(libsamplerate)
endif()