cmake_minimum_required(VERSION 3.15)

include(FetchContent)

# Set project name and language
project(python_rateshift LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Fetch libsamplerate if not present
set(BUILD_TESTING OFF CACHE BOOL "Disable libsamplerate test build")
set(LIBSAMPLERATE_VERSION 0.2.2 CACHE STRING "libsamplerate version")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate")
    message(STATUS "Fetching libsamplerate...")
    FetchContent_Declare(
        libsamplerate
        GIT_REPOSITORY https://github.com/libsndfile/libsamplerate
        GIT_TAG c96f5e3de9c4488f4e6c97f59f5245f22fda22f7 # 0.2.2
    )
    FetchContent_MakeAvailable(libsamplerate)
else()
    message(STATUS "Using existing libsamplerate in ${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate")
    add_subdirectory(libsamplerate)
endif()

# Define the Python module
pybind11_add_module(python_rateshift
    src/python_bindings.cpp
    src/rateshift.cpp
)

# Include directories
target_include_directories(python_rateshift
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/libsamplerate/include
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${Python_INCLUDE_DIRS}
)

# Compiler options
if(MSVC)
    target_compile_options(python_rateshift PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
else()
    target_compile_options(python_rateshift PRIVATE -O3 -Wall -Wextra)
endif()

# Compile definitions
target_compile_definitions(python_rateshift
    PUBLIC
        LIBSAMPLERATE_VERSION="${LIBSAMPLERATE_VERSION}"
    PRIVATE
        $<$<BOOL:${PACKAGE_VERSION_INFO}>:VERSION_INFO="${PACKAGE_VERSION_INFO}">
)

# Target properties
set_target_properties(python_rateshift
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "rateshift"
        LINKER_LANGUAGE CXX
)

# Link libraries
target_link_libraries(python_rateshift
    PRIVATE
        samplerate
        ${Python_LIBRARIES}
)

# Install rules
include(GNUInstallDirs)
install(TARGETS python_rateshift
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
