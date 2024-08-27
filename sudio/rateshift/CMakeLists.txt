cmake_minimum_required(VERSION 3.15)

# Set project name and language
project(python-rateshift LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CMake policies
cmake_policy(SET CMP0094 NEW)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python: ${Python_EXECUTABLE}")

# Add Python to prefix path
list(PREPEND CMAKE_PREFIX_PATH ${Python_PREFIX})

# Add external dependencies
add_subdirectory(external)

# Define the python module
pybind11_add_module(python-rateshift
    src/python_bindings.cpp
    src/rateshift.cpp
)

# Set include directories
target_include_directories(python-rateshift
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/libsamplerate/include
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# Set compiler options
if(MSVC)
    target_compile_options(python-rateshift PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
else()
    target_compile_options(python-rateshift PRIVATE -O3 -Wall -Wextra)
endif()

# Set compile definitions
target_compile_definitions(python-rateshift
    PUBLIC
        LIBSAMPLERATE_VERSION="${LIBSAMPLERATE_VERSION}"
    PRIVATE
        $<$<BOOL:${PACKAGE_VERSION_INFO}>:VERSION_INFO="${PACKAGE_VERSION_INFO}">
)

# Set target properties
set_target_properties(python-rateshift
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "rateshift"
        LINKER_LANGUAGE C
)

# Link libraries
target_link_libraries(python-rateshift
    PRIVATE
        samplerate
)

# Install rules
include(GNUInstallDirs)
install(TARGETS python-rateshift
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)