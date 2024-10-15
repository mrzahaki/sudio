# ./CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

project(sudio LANGUAGES CXX)


# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(C)
enable_language(CXX)

if (POLICY CMP0094)
    cmake_policy(SET CMP0094 NEW)
endif()

# Set Python registry and framework search preferences
if (NOT DEFINED Python_FIND_REGISTRY)
    set(Python_FIND_REGISTRY "LAST")
endif()
if (NOT DEFINED Python_FIND_FRAMEWORK)
    set(Python_FIND_FRAMEWORK "LAST")
endif()

# Find Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")

# Add Python to prefix path
list(PREPEND CMAKE_PREFIX_PATH ${Python_PREFIX})


include(FetchContent)
# Fetch pybind11 if not present
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.cache/pybind11")
    message(STATUS "Fetching pybind11...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG 5b0a6fc2017fcc176545afe3e09c9f9885283242 # 2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "Using existing pybind11 in ${CMAKE_CURRENT_SOURCE_DIR}/.cache/pybind11")
    add_subdirectory(".cache/pybind11")
endif()


include_directories(${PYTHON_INCLUDE_DIRS})
add_subdirectory(sudio/rateshift)
add_subdirectory(sudio/io)
